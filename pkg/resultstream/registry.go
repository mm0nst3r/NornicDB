package resultstream

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"hash"
	"io"
	"sync"
	"sync/atomic"
	"time"
)

const registryShardCount = 32

var (
	ErrInvalidQID   = errors.New("invalid result stream qid")
	ErrExpiredQID   = errors.New("result stream qid expired")
	ErrGoneQID      = errors.New("result stream qid no longer exists")
	ErrCapacity     = errors.New("result stream capacity exceeded")
	ErrDisabled     = errors.New("result stream continuation is disabled")
	ErrInvalidScope = errors.New("invalid result stream scope")
)

// Scope binds a stream to one authenticated owner and canonical database.
type Scope struct {
	Owner    string
	Database string
}

// Config bounds process-local durable result streams.
type Config struct {
	Disabled                 bool
	TTL                      time.Duration
	MaxStreams               int64
	MaxStreamsPerOwner       int64
	MaxPageSize              int
	MaxRetainedBytes         int64
	MaxRetainedBytesPerOwner int64
}

type registryEntry struct {
	scopeHash     [32]byte
	ownerHash     [32]byte
	expires       int64
	retainedBytes int64
	stream        Stream
	released      atomic.Bool
}

type ownerUsage struct {
	streams int64
	bytes   int64
}

type registryShard struct {
	mu      sync.RWMutex
	entries map[[16]byte]*registryEntry
}

// Registry owns signed qids and sharded process-local stream entries.
type Registry struct {
	config      Config
	disabled    bool
	secret      [32]byte
	instance    [8]byte
	shards      [registryShardCount]registryShard
	count       atomic.Int64
	retained    atomic.Int64
	closed      atomic.Bool
	scopeMAC    sync.Pool
	scopeInput  sync.Pool
	admissionMu sync.Mutex
	owners      map[[32]byte]ownerUsage
}

// NewRegistry constructs a registry with a random signing key and instance ID.
func NewRegistry(config Config) (*Registry, error) {
	if config.TTL < 0 || config.MaxStreams < 0 || config.MaxStreamsPerOwner < 0 || config.MaxPageSize < 0 ||
		config.MaxRetainedBytes < 0 || config.MaxRetainedBytesPerOwner < 0 {
		return nil, ErrCapacity
	}
	if config.TTL == 0 {
		config.TTL = 5 * time.Minute
	}
	if config.MaxStreams == 0 {
		config.MaxStreams = 1024
	}
	if config.MaxStreamsPerOwner == 0 {
		config.MaxStreamsPerOwner = 64
	}
	if config.MaxPageSize == 0 {
		config.MaxPageSize = 500
	}
	if config.MaxRetainedBytes == 0 {
		config.MaxRetainedBytes = 1 << 30
	}
	if config.MaxRetainedBytesPerOwner == 0 {
		config.MaxRetainedBytesPerOwner = 256 << 20
	}
	registry := &Registry{config: config, disabled: config.Disabled, owners: make(map[[32]byte]ownerUsage)}
	if _, err := io.ReadFull(rand.Reader, registry.secret[:]); err != nil {
		return nil, err
	}
	if _, err := io.ReadFull(rand.Reader, registry.instance[:]); err != nil {
		return nil, err
	}
	secret := registry.secret
	registry.scopeMAC.New = func() any { return hmac.New(sha256.New, secret[:]) }
	registry.scopeInput.New = func() any { return make([]byte, 0, 128) }
	for index := range registry.shards {
		registry.shards[index].entries = make(map[[16]byte]*registryEntry)
	}
	return registry, nil
}

// Start returns the first page and publishes the stream only when more rows are
// available.
func (r *Registry) Start(ctx context.Context, scope Scope, stream Stream, n int) (*Page, error) {
	if err := r.validate(scope, n); err != nil {
		return nil, err
	}
	page, err := stream.Pull(ctx, 0, n)
	if err != nil {
		_ = stream.Close()
		return nil, err
	}
	if !page.HasMore {
		_ = stream.Close()
		return page, nil
	}
	retainedBytes := int64(0)
	if reporter, ok := stream.(RetainedBytesReporter); ok {
		retainedBytes = reporter.RetainedBytes()
		if retainedBytes < 0 {
			_ = stream.Close()
			return nil, ErrCapacity
		}
	}
	ownerHash := r.ownerDigest(scope.Owner)
	if !r.reserve(ownerHash, retainedBytes) {
		_ = stream.Close()
		return nil, ErrCapacity
	}

	entry := &registryEntry{
		scopeHash:     r.scopeDigest(scope),
		ownerHash:     ownerHash,
		expires:       time.Now().Add(r.config.TTL).Unix(),
		retainedBytes: retainedBytes,
		stream:        stream,
	}
	var streamID [16]byte
	inserted := false
	for attempt := 0; attempt < 4; attempt++ {
		if _, err := io.ReadFull(rand.Reader, streamID[:]); err != nil {
			break
		}
		shard := r.shard(streamID)
		shard.mu.Lock()
		if _, exists := shard.entries[streamID]; !exists && !r.closed.Load() {
			shard.entries[streamID] = entry
			inserted = true
		}
		shard.mu.Unlock()
		if inserted {
			break
		}
	}
	if !inserted {
		r.release(entry)
		_ = stream.Close()
		if r.closed.Load() {
			return nil, ErrClosed
		}
		return nil, ErrCapacity
	}
	page.QID = encodeToken(r.secret, r.instance, streamID, page.Next, entry.expires)
	page.ExpiresAt = time.Unix(entry.expires, 0).UTC()
	return page, nil
}

// Pull resolves a signed qid and invokes the stream without holding a registry
// shard lock.
func (r *Registry) Pull(ctx context.Context, scope Scope, qid string, n int) (*Page, error) {
	if r.disabled {
		return nil, ErrDisabled
	}
	if err := r.validate(scope, n); err != nil {
		return nil, err
	}
	token, err := decodeToken(r.secret, r.instance, qid)
	if err != nil {
		return nil, err
	}
	if time.Now().Unix() >= token.expires {
		r.remove(token.streamID, nil)
		return nil, ErrExpiredQID
	}
	entry := r.lookup(token.streamID)
	wantScope := r.scopeDigest(scope)
	if entry == nil || entry.released.Load() {
		return nil, ErrGoneQID
	}
	if entry.expires != token.expires || !hmac.Equal(entry.scopeHash[:], wantScope[:]) {
		return nil, ErrInvalidQID
	}
	page, err := entry.stream.Pull(ctx, token.position, n)
	if err != nil {
		return nil, err
	}
	if page.HasMore {
		page.QID = encodeToken(r.secret, r.instance, token.streamID, page.Next, entry.expires)
	}
	page.ExpiresAt = time.Unix(entry.expires, 0).UTC()
	return page, nil
}

// Discard releases the complete stream addressed by any of its qids.
func (r *Registry) Discard(scope Scope, qid string) error {
	if r.disabled {
		return ErrDisabled
	}
	if scope.Owner == "" || scope.Database == "" {
		return ErrInvalidScope
	}
	token, err := decodeToken(r.secret, r.instance, qid)
	if err != nil {
		return err
	}
	if time.Now().Unix() >= token.expires {
		r.remove(token.streamID, nil)
		return ErrExpiredQID
	}
	entry := r.lookup(token.streamID)
	wantScope := r.scopeDigest(scope)
	if entry == nil || entry.released.Load() {
		return ErrGoneQID
	}
	if entry.expires != token.expires || !hmac.Equal(entry.scopeHash[:], wantScope[:]) {
		return ErrInvalidQID
	}
	r.remove(token.streamID, entry)
	return nil
}

func (r *Registry) validate(scope Scope, n int) error {
	if r.disabled {
		return ErrDisabled
	}
	if r.closed.Load() {
		return ErrClosed
	}
	if scope.Owner == "" || scope.Database == "" {
		return ErrInvalidScope
	}
	if n <= 0 || n > r.config.MaxPageSize {
		return ErrInvalidPageSize
	}
	return nil
}

func (r *Registry) lookup(id [16]byte) *registryEntry {
	shard := r.shard(id)
	shard.mu.RLock()
	entry := shard.entries[id]
	shard.mu.RUnlock()
	return entry
}

func (r *Registry) remove(id [16]byte, expected *registryEntry) {
	shard := r.shard(id)
	shard.mu.Lock()
	entry := shard.entries[id]
	if entry != nil && (expected == nil || expected == entry) {
		delete(shard.entries, id)
		entry.released.Store(true)
	}
	shard.mu.Unlock()
	if entry != nil && (expected == nil || expected == entry) {
		r.release(entry)
		_ = entry.stream.Close()
	}
}

func (r *Registry) reserve(owner [32]byte, bytes int64) bool {
	r.admissionMu.Lock()
	defer r.admissionMu.Unlock()
	usage := r.owners[owner]
	if r.count.Load() >= r.config.MaxStreams || usage.streams >= r.config.MaxStreamsPerOwner ||
		r.retained.Load()+bytes > r.config.MaxRetainedBytes || usage.bytes+bytes > r.config.MaxRetainedBytesPerOwner {
		return false
	}
	usage.streams++
	usage.bytes += bytes
	r.owners[owner] = usage
	r.count.Add(1)
	r.retained.Add(bytes)
	return true
}

func (r *Registry) release(entry *registryEntry) {
	r.admissionMu.Lock()
	usage := r.owners[entry.ownerHash]
	usage.streams--
	usage.bytes -= entry.retainedBytes
	if usage.streams == 0 {
		delete(r.owners, entry.ownerHash)
	} else {
		r.owners[entry.ownerHash] = usage
	}
	r.count.Add(-1)
	r.retained.Add(-entry.retainedBytes)
	r.admissionMu.Unlock()
}

func (r *Registry) shard(id [16]byte) *registryShard {
	return &r.shards[id[0]&byte(registryShardCount-1)]
}

func scopeDigest(secret [32]byte, scope Scope) [32]byte {
	digest := hmac.New(sha256.New, secret[:])
	_, _ = digest.Write([]byte(scope.Owner))
	_, _ = digest.Write([]byte{0})
	_, _ = digest.Write([]byte(scope.Database))
	var out [32]byte
	copy(out[:], digest.Sum(nil))
	return out
}

func (r *Registry) scopeDigest(scope Scope) [32]byte {
	digest := r.scopeMAC.Get().(hash.Hash)
	digest.Reset()
	input := r.scopeInput.Get().([]byte)[:0]
	input = append(input, scope.Owner...)
	input = append(input, 0)
	input = append(input, scope.Database...)
	_, _ = digest.Write(input)
	var out [32]byte
	digest.Sum(out[:0])
	clear(input)
	r.scopeInput.Put(input[:0])
	r.scopeMAC.Put(digest)
	return out
}

func (r *Registry) ownerDigest(owner string) [32]byte {
	return scopeDigest(r.secret, Scope{Owner: owner})
}

// Close releases every retained stream and rejects future operations.
func (r *Registry) Close() {
	if !r.closed.CompareAndSwap(false, true) {
		return
	}
	for index := range r.shards {
		shard := &r.shards[index]
		shard.mu.Lock()
		entries := shard.entries
		shard.entries = make(map[[16]byte]*registryEntry)
		shard.mu.Unlock()
		for _, entry := range entries {
			entry.released.Store(true)
			r.release(entry)
			_ = entry.stream.Close()
		}
	}
}
