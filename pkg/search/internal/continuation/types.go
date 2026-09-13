// Package continuation owns bounded, process-local search populations and opaque
// replayable cursors. It deliberately has no dependency on a retrieval backend.
package continuation

import (
	"encoding/json"
	"errors"
	"time"
)

// Mode declares which population the cursor enumerates.
type Mode string

const (
	// RankedOnly enumerates the single bounded retrieval result, not the corpus.
	RankedOnly Mode = "ranked"
	// RankedThenID enumerates a ranked prefix, then all other eligible members.
	RankedThenID Mode = "ranked_then_id"
	// IDOnly enumerates the complete eligible population without retrieval.
	IDOnly Mode = "id"
)

// Phase distinguishes relevance-ranked hits from an unscored catalogue tail.
type Phase string

const (
	RankedPhase           Phase = "ranked"
	CatalogPhase          Phase = "catalog"
	MoreResults                 = "more_results"
	CandidatePoolComplete       = "candidate_pool_exhausted"
	CollectionComplete          = "eligible_population_exhausted"
)

var (
	ErrInvalidRequest = errors.New("search continuation: invalid request")
	ErrInvalidCursor  = errors.New("search continuation: invalid cursor")
	ErrCursorMismatch = errors.New("search continuation: query or scope changed")
	ErrCursorExpired  = errors.New("search continuation: cursor expired or released; restart the query")
	ErrInvalidated    = errors.New("search continuation: indexed data or policy changed; restart the query")
	ErrCapacity       = errors.New("search continuation: resource limit exceeded")
	ErrClosed         = errors.New("search continuation: service closed")
	ErrTicketClosed   = errors.New("search continuation: build already finished")
)

// Hit is an immutable value descriptor; it never retains a node or embedding.
// Score is meaningful only in RankedPhase. Catalogue hits are unscored, not
// low-relevance matches. GroupKey is the distinct parent key when grouping.
type Hit struct {
	// Metadata holds copied explanatory JSON fields from retrieval. JSON output
	// flattens them without permitting overrides of this descriptor's fields.
	Metadata   json.RawMessage `json:"-"`
	ID         string          `json:"id"`
	GroupKey   string          `json:"group_key,omitempty"`
	Phase      Phase           `json:"phase"`
	Score      float64         `json:"score"`
	Similarity float64         `json:"similarity,omitempty"`
	RRFScore   float64         `json:"rrf_score,omitempty"`
	VectorRank int             `json:"vector_rank"`
	BM25Rank   int             `json:"bm25_rank"`
}

// Population is the fixed order selected by one initial request.
type Population struct {
	Metadata             json.RawMessage
	TotalCandidates      int
	FallbackTriggered    bool
	VectorStopReason     string
	VectorCandidateLimit int
	BM25StopReason       string
	BM25CandidateLimit   int
	Hits                 []Hit
	Mode                 Mode
	Grouped              bool
	RankedCount          int
	SearchMethod         string
	CandidateLimit       int
}

// Page explicitly distinguishes selected-pool exhaustion from corpus exhaustion.
// EligibleCount is unknown (nil) for RankedOnly, even for an empty ANN result.
// Position is the zero-based starting position in this population.
type Page struct {
	// Metadata is the initial retrieval's copied diagnostic JSON object. It is
	// repeated on every page; JSON output preserves the original field names.
	Metadata             json.RawMessage `json:"-"`
	TotalCandidates      int             `json:"total_candidates"`
	FallbackTriggered    bool            `json:"fallback_triggered"`
	VectorStopReason     string          `json:"vector_stop_reason,omitempty"`
	VectorCandidateLimit int             `json:"vector_candidate_limit,omitempty"`
	BM25StopReason       string          `json:"bm25_stop_reason,omitempty"`
	BM25CandidateLimit   int             `json:"bm25_candidate_limit,omitempty"`
	Results              []Hit           `json:"results"`
	NextCursor           string          `json:"next_cursor,omitempty"`
	Returned             int             `json:"returned"`
	Position             int             `json:"position"`
	Total                int             `json:"total"`
	RankedCount          int             `json:"ranked_count"`
	EligibleCount        *int            `json:"eligible_count,omitempty"`
	Mode                 Mode            `json:"mode"`
	Grouped              bool            `json:"grouped"`
	Population           string          `json:"population"`
	SearchMethod         string          `json:"search_method"`
	CandidateLimit       int             `json:"candidate_limit"`
	RankedPoolExhausted  bool            `json:"ranked_pool_exhausted"`
	Exhausted            bool            `json:"exhausted"`
	CollectionExhausted  bool            `json:"collection_exhausted"`
	Completion           string          `json:"completion"`
	ExpiresAt            time.Time       `json:"expires_at"`
}

// Config bounds retained state and simultaneous initial materialisations.
// Byte limits account conservatively for descriptors and strings, not total Go
// process RSS. There are no background workers. Expired state is reclaimed on
// admission/continuation, mutation, reconfiguration, or Close.
type Config struct {
	TTL             time.Duration
	MaxSessions     int
	MaxResults      int
	MaxBytes        int64
	MaxBuildBytes   int64
	MaxBuilds       int
	MaxPageSize     int
	MaxCandidates   int
	MaxScannedNodes int
}

// DefaultConfig is deliberately suitable for a 20,000-asset catalogue.
func DefaultConfig() Config {
	return Config{
		TTL: 5 * time.Minute, MaxSessions: 32, MaxResults: 100000,
		MaxBytes: 64 << 20, MaxBuildBytes: 32 << 20, MaxBuilds: 1,
		MaxPageSize: 500, MaxCandidates: 5000, MaxScannedNodes: 1000000,
	}
}

// NormalizeConfig fills zero fields with defaults and rejects negative limits.
func NormalizeConfig(c Config) (Config, error) {
	if c.TTL < 0 || c.MaxSessions < 0 || c.MaxResults < 0 || c.MaxBytes < 0 ||
		c.MaxBuildBytes < 0 || c.MaxBuilds < 0 || c.MaxPageSize < 0 ||
		c.MaxCandidates < 0 || c.MaxScannedNodes < 0 {
		return Config{}, ErrInvalidRequest
	}
	d := DefaultConfig()
	if c.TTL == 0 {
		c.TTL = d.TTL
	}
	if c.MaxSessions == 0 {
		c.MaxSessions = d.MaxSessions
	}
	if c.MaxResults == 0 {
		c.MaxResults = d.MaxResults
	}
	if c.MaxBytes == 0 {
		c.MaxBytes = d.MaxBytes
	}
	if c.MaxBuildBytes == 0 {
		c.MaxBuildBytes = d.MaxBuildBytes
	}
	if c.MaxBuilds == 0 {
		c.MaxBuilds = d.MaxBuilds
	}
	if c.MaxPageSize == 0 {
		c.MaxPageSize = d.MaxPageSize
	}
	if c.MaxCandidates == 0 {
		c.MaxCandidates = d.MaxCandidates
	}
	if c.MaxScannedNodes == 0 {
		c.MaxScannedNodes = d.MaxScannedNodes
	}
	return c, nil
}

func validMode(m Mode) bool { return m == RankedOnly || m == RankedThenID || m == IDOnly }

// Descriptor accounting includes space for map/slice overhead during builds.
const descriptorBytes int64 = 256

func hitBytes(h Hit) int64 {
	return descriptorBytes + int64(len(h.ID)) + int64(len(h.GroupKey)) + int64(len(h.Metadata))
}
