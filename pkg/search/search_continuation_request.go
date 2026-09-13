package search

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// request bytes bound hashing/copy work as well as retained identifier strings.
// This is a safety ceiling, not a recommendation for large metadata filters.
const maxSearchPageRequestBytes = 1 << 20

type resolvedSearchPageRequest struct {
	query       string
	embedding   []float32
	options     SearchOptions
	page        SearchPageOptions
	fingerprint [32]byte
}

func (s *Service) resolveSearchPageRequest(query string, embedding []float32, opts *SearchOptions, paging *SearchPageOptions) (*resolvedSearchPageRequest, error) {
	if paging == nil || paging.Scope == "" {
		return nil, fmt.Errorf("a trusted scope is required: %w", ErrSearchPageRequest)
	}
	page := *paging
	if page.Mode == "" {
		page.Mode = SearchPageRanked
	}
	if page.Mode != SearchPageID && page.Mode != SearchPageRanked && page.Mode != SearchPageRankedThenID {
		return nil, ErrSearchPageRequest
	}
	if page.PageSize == 0 {
		page.PageSize = 50
	}
	policy := s.continuation.store.Config()
	policy.MaxCandidates = min(policy.MaxCandidates, MaxCandidates)
	if page.PageSize < 1 || page.PageSize > policy.MaxPageSize {
		return nil, ErrSearchPageRequest
	}
	if opts == nil {
		opts = DefaultSearchOptions()
	}
	if opts.MMREnabled {
		return nil, fmt.Errorf("MMR is not a score-ordered population: %w", ErrSearchPageRequest)
	}
	// Validate lengths before allocating copies. JSON encoding also rejects all
	// NaN/Inf floating-point settings and embeddings, avoiding ambiguous hashes.
	bytes := len(query) + len(page.GroupBy) + len(page.Scope)
	if !utf8.ValidString(query) || !utf8.ValidString(page.GroupBy) || !utf8.ValidString(page.Scope) || len(embedding) > maxSearchPageRequestBytes/4 {
		return nil, ErrSearchPageRequest
	}
	bytes += len(embedding) * 4
	if bytes > maxSearchPageRequestBytes {
		return nil, ErrSearchPageRequest
	}
	for _, kind := range opts.Types {
		if !utf8.ValidString(kind) || len(kind) > maxSearchPageRequestBytes-bytes {
			return nil, ErrSearchPageRequest
		}
		bytes += len(kind) + 16
	}
	for key, values := range opts.Filters {
		if !utf8.ValidString(key) || len(key) > maxSearchPageRequestBytes-bytes {
			return nil, ErrSearchPageRequest
		}
		bytes += len(key) + 16
		for _, value := range values {
			if !utf8.ValidString(value) || len(value) > maxSearchPageRequestBytes-bytes {
				return nil, ErrSearchPageRequest
			}
			bytes += len(value) + 16
		}
	}
	if bytes > maxSearchPageRequestBytes {
		return nil, ErrSearchPageRequest
	}
	o := *opts
	if page.Mode != SearchPageID {
		// Ratios beyond the hard candidate cap cannot retrieve more useful
		// candidates, and must not overflow float-to-int budget arithmetic.
		for _, ratio := range []float64{o.InitialOverfetchRatio, o.MaxOverfetchRatio, o.OverfetchGrowthFactor} {
			if ratio < 0 || ratio > float64(MaxCandidates) {
				return nil, ErrSearchPageRequest
			}
		}
		if o.Limit <= 0 {
			o.Limit = DefaultSearchOptions().Limit
		}
		if o.Limit > policy.MaxCandidates || o.CandidateTarget < 0 || o.CandidateTarget > policy.MaxCandidates || o.MaxCandidateLimit < 0 {
			return nil, ErrSearchPageRequest
		}
		if o.MaxCandidateLimit == 0 || o.MaxCandidateLimit > policy.MaxCandidates {
			o.MaxCandidateLimit = policy.MaxCandidates
		}
		if (o.RerankEnabled && o.RerankTopK > policy.MaxCandidates) || o.RerankTopK < 0 {
			return nil, ErrSearchPageRequest
		}
	}
	// Search mutates MinSimilarity on its options; never give it caller-owned
	// pointers, slices, maps, or the struct bound into this request fingerprint.
	minSimilarity := *s.resolveMinSimilarity(&o)
	o.MinSimilarity = &minSimilarity
	if o.FallbackEnabled != nil {
		v := *o.FallbackEnabled
		o.FallbackEnabled = &v
	}
	o.Types = make([]string, len(opts.Types))
	for i, kind := range opts.Types {
		o.Types[i] = strings.ToLower(kind)
	}
	o.Types = canonicalSearchPageStrings(o.Types)
	o.Filters = make(map[string][]string, len(opts.Filters))
	for key, values := range opts.Filters {
		if len(values) == 0 {
			continue
		} // same semantics as nodeMatchesFilters
		o.Filters[key] = canonicalSearchPageStrings(append([]string(nil), values...))
	}
	r := &resolvedSearchPageRequest{query: query, embedding: append([]float32(nil), embedding...), options: o, page: page}
	// JSON preserves delimiters and escapes; sorted map keys plus normalized
	// sets avoid the collisions of concatenated query/cache-key strings.
	encoded, err := json.Marshal(struct {
		Version        int
		Query          string
		Embedding      []float32
		Options        SearchOptions
		Mode           SearchPageMode
		GroupBy, Scope string
	}{1, query, r.embedding, o, page.Mode, page.GroupBy, page.Scope})
	if err != nil {
		return nil, fmt.Errorf("non-serializable search request: %w", ErrSearchPageRequest)
	}
	if len(encoded) > maxSearchPageRequestBytes {
		return nil, ErrSearchPageRequest
	}
	r.fingerprint = sha256.Sum256(encoded)
	return r, nil
}

func canonicalSearchPageStrings(values []string) []string {
	sort.Strings(values)
	out := values[:0]
	for _, value := range values {
		if len(out) == 0 || out[len(out)-1] != value {
			out = append(out, value)
		}
	}
	return out
}
