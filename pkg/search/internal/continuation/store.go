package continuation

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/rand"
	"fmt"
	"io"
	"math"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

type session struct {
	id          [16]byte
	generation  uint64
	fingerprint [32]byte
	population  Population
	expires     time.Time
	bytes       int64
}

// Store's zero value is ready to use. Do not copy a Store after first use.
// It is bound to one service/database and has its own random signing key.
type Store struct {
	mu          sync.Mutex
	config      Config
	initialized bool
	closed      bool
	generation  uint64
	writers     int
	builds      int
	entries     map[[16]byte]*session
	bytes       int64
	secret      [32]byte
	keyReady    bool
	// These private seams keep time/entropy error tests deterministic.
	clock   func() time.Time
	entropy io.Reader
}

func (s *Store) initLocked() {
	if !s.initialized {
		s.config = DefaultConfig()
		s.initialized = true
	}
}
func (s *Store) now() time.Time {
	if s.clock != nil {
		return s.clock()
	}
	return time.Now()
}
func (s *Store) random() io.Reader {
	if s.entropy != nil {
		return s.entropy
	}
	return rand.Reader
}

// Config returns a copy of the effective resource policy.
func (s *Store) Config() Config {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initLocked()
	return s.config
}

// Configure changes limits and invalidates outstanding cursors/builds. It does
// not silently evict one user's live cursor to admit another user's query.
func (s *Store) Configure(config Config) error {
	c, err := NormalizeConfig(config)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	s.config = c
	s.initialized = true
	s.invalidateLocked()
	return nil
}

func (s *Store) invalidateLocked() {
	s.generation++
	s.entries = nil
	s.bytes = 0
}

// BeginMutation invalidates before a write starts and after it finishes. The
// returned release function is idempotent. Nesting is supported (index builds
// can call indexing methods). No retrieval may be admitted across a mutation.
func (s *Store) BeginMutation() func() {
	s.mu.Lock()
	s.writers++
	s.invalidateLocked()
	s.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			s.mu.Lock()
			defer s.mu.Unlock()
			s.writers--
			s.invalidateLocked()
		})
	}
}

// Close invalidates cursors and permanently closes admission.
func (s *Store) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	s.invalidateLocked()
}

// Ticket reserves one initial-build slot and records the pre-retrieval epoch.
// Always defer Abort; Commit also releases the slot on success or failure.
type Ticket struct {
	store      *Store
	generation uint64
	config     Config
	finished   bool // guarded by store.mu
}

// Config returns this build's immutable resource policy.
func (t *Ticket) Config() Config { return t.config }

// Start admits an initial retrieval before allocating its population.
func (s *Store) Start(ctx context.Context) (*Ticket, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initLocked()
	if s.closed {
		return nil, ErrClosed
	}
	if s.writers > 0 {
		return nil, ErrInvalidated
	}
	s.pruneLocked(s.now())
	if s.builds >= s.config.MaxBuilds {
		return nil, ErrCapacity
	}
	if !s.keyReady {
		if _, err := io.ReadFull(s.random(), s.secret[:]); err != nil {
			return nil, fmt.Errorf("continuation signing key: %w", err)
		}
		s.keyReady = true
	}
	s.builds++
	return &Ticket{store: s, generation: s.generation, config: s.config}, nil
}

// Abort releases a build slot exactly once; it never deletes an admitted page.
func (t *Ticket) Abort() {
	if t == nil || t.store == nil {
		return
	}
	s := t.store
	s.mu.Lock()
	defer s.mu.Unlock()
	t.finishLocked()
}
func (t *Ticket) finishLocked() {
	if !t.finished {
		t.finished = true
		t.store.builds--
	}
}

// Check detects a mutation while a potentially long initial scan is running.
func (t *Ticket) Check(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s := t.store
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	if t.finished {
		return ErrTicketClosed
	}
	if s.writers > 0 || s.generation != t.generation {
		return ErrInvalidated
	}
	return nil
}

// Commit publishes only a complete population from an unchanged epoch. It
// copies input descriptors so neither a builder nor a page recipient can edit
// the cached population. An oversized exact population fails, never truncates.
func (t *Ticket) Commit(ctx context.Context, fingerprint [32]byte, population Population, pageSize int) (*Page, error) {
	s := t.store
	s.mu.Lock()
	defer s.mu.Unlock()
	if t.finished {
		return nil, ErrTicketClosed
	}
	defer t.finishLocked()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.closed {
		return nil, ErrClosed
	}
	if s.writers > 0 || s.generation != t.generation {
		return nil, ErrInvalidated
	}
	if pageSize <= 0 || pageSize > s.config.MaxPageSize {
		return nil, ErrInvalidRequest
	}
	owned, bytes, err := clonePopulation(ctx, population, t.config)
	if err != nil {
		return nil, err
	}
	now := s.now()
	s.pruneLocked(now)
	entry := &session{generation: t.generation, fingerprint: fingerprint, population: owned, expires: now.Add(t.config.TTL), bytes: bytes}
	// Complete one-page responses need no retained session at all.
	if len(owned.Hits) <= pageSize {
		return s.pageLocked(entry, 0, pageSize), nil
	}
	if len(s.entries) >= s.config.MaxSessions || bytes > s.config.MaxBytes-s.bytes {
		return nil, ErrCapacity
	}
	// A bounded retry avoids accidental replacement even with a faulty RNG.
	unique := false
	for attempt := 0; attempt < 4; attempt++ {
		if _, err := io.ReadFull(s.random(), entry.id[:]); err != nil {
			return nil, fmt.Errorf("continuation session ID: %w", err)
		}
		if _, exists := s.entries[entry.id]; !exists {
			unique = true
			break
		}
	}
	if !unique {
		return nil, fmt.Errorf("continuation session ID collision: %w", ErrCapacity)
	}
	if s.entries == nil {
		s.entries = make(map[[16]byte]*session)
	}
	s.entries[entry.id] = entry
	s.bytes += bytes
	return s.pageLocked(entry, 0, pageSize), nil
}

func clonePopulation(ctx context.Context, p Population, c Config) (Population, int64, error) {
	if !validMetadata(p.Metadata) || !validMode(p.Mode) || p.RankedCount < 0 || p.RankedCount > len(p.Hits) || p.CandidateLimit < 0 ||
		(p.Mode == RankedOnly && p.RankedCount != len(p.Hits)) || (p.Mode == IDOnly && p.RankedCount != 0) {
		return Population{}, 0, ErrInvalidRequest
	}
	if len(p.Hits) > c.MaxResults {
		return Population{}, 0, ErrCapacity
	}
	retainedBytes := int64(len(p.SearchMethod)) + int64(len(p.Metadata)) + 256
	if retainedBytes > c.MaxBytes || retainedBytes > c.MaxBuildBytes {
		return Population{}, 0, ErrCapacity
	}
	out := p
	out.Metadata = bytes.Clone(p.Metadata)
	out.SearchMethod = strings.Clone(p.SearchMethod)
	out.Hits = make([]Hit, 0, len(p.Hits))
	seenIDs := make(map[string]bool, len(p.Hits))
	seenGroups := make(map[string]bool)
	for i, h := range p.Hits {
		if err := ctx.Err(); err != nil {
			return Population{}, 0, err
		}
		expectedPhase := CatalogPhase
		if i < p.RankedCount {
			expectedPhase = RankedPhase
		}
		if h.ID == "" || !utf8.ValidString(h.ID) || !utf8.ValidString(h.GroupKey) ||
			h.Phase != expectedPhase || seenIDs[h.ID] ||
			(p.Grouped && (h.GroupKey == "" || seenGroups[h.GroupKey])) || (!p.Grouped && h.GroupKey != "") ||
			!finiteHit(h) || !validMetadata(h.Metadata) {
			return Population{}, 0, ErrInvalidRequest
		}
		size := hitBytes(h)
		if size > c.MaxBytes-retainedBytes || size > c.MaxBuildBytes-retainedBytes {
			return Population{}, 0, ErrCapacity
		}
		retainedBytes += size
		seenIDs[h.ID] = true
		if p.Grouped {
			seenGroups[h.GroupKey] = true
		}
		h.ID = strings.Clone(h.ID)
		h.GroupKey = strings.Clone(h.GroupKey)
		h.Metadata = bytes.Clone(h.Metadata)
		out.Hits = append(out.Hits, h)
	}
	return out, retainedBytes, nil
}

func finiteHit(h Hit) bool {
	for _, x := range []float64{h.Score, h.Similarity, h.RRFScore} {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

func (s *Store) pruneLocked(now time.Time) {
	for id, entry := range s.entries {
		if !now.Before(entry.expires) {
			s.bytes -= entry.bytes
			delete(s.entries, id)
		}
	}
}

// Continue returns the slice at a signed offset without rerunning retrieval,
// scanning storage, or extending TTL. Repeating a cursor is idempotent.
func (s *Store) Continue(ctx context.Context, cursor string, fingerprint [32]byte, pageSize int) (*Page, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initLocked()
	if s.closed {
		return nil, ErrClosed
	}
	if pageSize <= 0 || pageSize > s.config.MaxPageSize {
		return nil, ErrInvalidRequest
	}
	id, generation, position, err := s.decodeCursorLocked(cursor)
	if err != nil {
		return nil, err
	}
	if s.writers > 0 || generation != s.generation {
		return nil, ErrInvalidated
	}
	s.pruneLocked(s.now())
	entry, ok := s.entries[id]
	if !ok {
		return nil, ErrCursorExpired
	}
	if !hmac.Equal(entry.fingerprint[:], fingerprint[:]) {
		return nil, ErrCursorMismatch
	}
	if position >= uint64(len(entry.population.Hits)) {
		return nil, ErrInvalidCursor
	}
	return s.pageLocked(entry, int(position), pageSize), nil
}

// Release explicitly frees a live cursor's population. Releasing any cursor in
// a session also ends replay for its other pages. Authentication is identical
// to Continue. Expiry already makes an additional release unnecessary.
func (s *Store) Release(cursor string, fingerprint [32]byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	id, generation, _, err := s.decodeCursorLocked(cursor)
	if err != nil {
		return err
	}
	if s.writers > 0 || generation != s.generation {
		return ErrInvalidated
	}
	s.pruneLocked(s.now())
	entry, ok := s.entries[id]
	if !ok {
		return ErrCursorExpired
	}
	if !hmac.Equal(entry.fingerprint[:], fingerprint[:]) {
		return ErrCursorMismatch
	}
	delete(s.entries, id)
	s.bytes -= entry.bytes
	return nil
}

func (s *Store) pageLocked(entry *session, position, pageSize int) *Page {
	p := entry.population
	end := position + min(pageSize, len(p.Hits)-position)
	results := make([]Hit, end-position)
	copy(results, p.Hits[position:end])
	for i := range results {
		results[i].Metadata = bytes.Clone(results[i].Metadata)
	}
	done := end == len(p.Hits)
	page := &Page{Results: results, Returned: len(results), Position: position, Total: len(p.Hits),
		Metadata:        bytes.Clone(p.Metadata),
		TotalCandidates: p.TotalCandidates, FallbackTriggered: p.FallbackTriggered,
		VectorStopReason: p.VectorStopReason, VectorCandidateLimit: p.VectorCandidateLimit,
		BM25StopReason: p.BM25StopReason, BM25CandidateLimit: p.BM25CandidateLimit,
		RankedCount: p.RankedCount, Mode: p.Mode, Grouped: p.Grouped, SearchMethod: p.SearchMethod,
		CandidateLimit: p.CandidateLimit, RankedPoolExhausted: end >= p.RankedCount,
		Exhausted: done, CollectionExhausted: done && p.Mode != RankedOnly, Completion: MoreResults,
		ExpiresAt: entry.expires, Population: "ranked_candidates"}
	if p.Mode != RankedOnly {
		total := len(p.Hits)
		page.EligibleCount = &total
		page.Population = "filtered_collection"
	}
	if done {
		page.Completion = CandidatePoolComplete
		if page.CollectionExhausted {
			page.Completion = CollectionComplete
		}
	} else {
		page.NextCursor = s.encodeCursorLocked(entry, uint64(end))
	}
	return page
}
