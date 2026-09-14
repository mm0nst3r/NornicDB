package search

import (
	"cmp"
	"context"
	"slices"
)

const (
	maxTextQueryChunks     = 32
	outerRRFK              = 60.0
	minChunkCandidateLimit = 10
	maxChunkCandidateLimit = 100
)

// ChunkQueryFunc splits a text query into embedding-safe chunks.
type ChunkQueryFunc func(ctx context.Context, query string) ([]string, error)

// EmbedQueryFunc embeds one query chunk. A nil or empty vector is unavailable.
type EmbedQueryFunc func(ctx context.Context, query string) ([]float32, error)

// SearchQueryFunc searches one query and optional vector.
type SearchQueryFunc func(ctx context.Context, query string, embedding []float32, opts *SearchOptions) (*SearchResponse, error)

// ChunkedSearchErrorPolicy identifies adapter-specific errors that must not fall back.
type ChunkedSearchErrorPolicy struct {
	FatalEmbeddingError func(error) bool
	FatalSearchError    func(error) bool
}

// SearchTextChunks applies the canonical text-search semantics used by all transports.
// Multi-chunk queries are searched independently and fused with an outer RRF pass.
func SearchTextChunks(
	ctx context.Context,
	query string,
	opts *SearchOptions,
	chunkQuery ChunkQueryFunc,
	embedQuery EmbedQueryFunc,
	searchQuery SearchQueryFunc,
) (*SearchResponse, error) {
	return SearchTextChunksWithErrorPolicy(ctx, query, opts, chunkQuery, embedQuery, searchQuery, ChunkedSearchErrorPolicy{})
}

// SearchTextChunksWithErrorPolicy applies SearchTextChunks while allowing an adapter
// to designate errors that must be returned instead of triggering BM25 fallback.
func SearchTextChunksWithErrorPolicy(
	ctx context.Context,
	query string,
	opts *SearchOptions,
	chunkQuery ChunkQueryFunc,
	embedQuery EmbedQueryFunc,
	searchQuery SearchQueryFunc,
	errorPolicy ChunkedSearchErrorPolicy,
) (*SearchResponse, error) {
	if opts == nil {
		opts = DefaultSearchOptions()
	}
	if embedQuery == nil {
		return searchQuery(ctx, query, nil, opts)
	}

	chunks := []string{query}
	if chunkQuery != nil {
		var err error
		chunks, err = chunkQuery(ctx, query)
		if err != nil {
			return nil, err
		}
	}
	if len(chunks) > maxTextQueryChunks {
		chunks = chunks[:maxTextQueryChunks]
	}
	if len(chunks) <= 1 {
		embedding, err := embedQuery(ctx, query)
		if err != nil && isFatalChunkedSearchError(errorPolicy.FatalEmbeddingError, err) {
			return nil, err
		}
		if err == nil && len(embedding) > 0 {
			response, searchErr := searchQuery(ctx, query, embedding, opts)
			if searchErr != nil && isFatalChunkedSearchError(errorPolicy.FatalSearchError, searchErr) {
				return nil, searchErr
			}
			if searchErr == nil && response != nil {
				return response, nil
			}
		}
		return searchQuery(ctx, query, nil, opts)
	}

	type fusedResult struct {
		best  *SearchResult
		score float64
	}

	chunkOpts := *opts
	chunkOpts.Limit = chunkCandidateLimit(opts.Limit)
	if opts.continuation {
		// Continued queries deepen every chunk, rather than repeatedly searching
		// the same one-shot top-100 prefix.
		chunkOpts.Limit = max(chunkOpts.Limit, min(opts.Limit, MaxCandidates))
	}
	exhausted := true
	var (
		fusedIndexes map[string]int
		fused        []fusedResult
	)
	for _, chunk := range chunks {
		embedding, err := embedQuery(ctx, chunk)
		if err != nil && isFatalChunkedSearchError(errorPolicy.FatalEmbeddingError, err) {
			return nil, err
		}
		if err != nil || len(embedding) == 0 {
			exhausted = false
			continue
		}
		response, err := searchQuery(ctx, chunk, embedding, &chunkOpts)
		if err != nil && isFatalChunkedSearchError(errorPolicy.FatalSearchError, err) {
			return nil, err
		}
		if err != nil || response == nil {
			exhausted = false
			continue
		}
		exhausted = exhausted && response.RetrievalExhausted
		if fusedIndexes == nil && len(response.Results) > 0 {
			candidatesPerChunk := chunkOpts.Limit
			if len(response.Results) > candidatesPerChunk {
				candidatesPerChunk = len(response.Results)
			}
			capacity := candidatesPerChunk * len(chunks)
			fusedIndexes = make(map[string]int, capacity)
			fused = make([]fusedResult, 0, capacity)
		}
		for rank := range response.Results {
			result := &response.Results[rank]
			id := string(result.NodeID)
			if id == "" {
				id = result.ID
			}
			index, exists := fusedIndexes[id]
			if !exists {
				fused = append(fused, fusedResult{best: result})
				index = len(fused)
				fusedIndexes[id] = index
			}
			fusedResult := &fused[index-1]
			if exists && result.Score > fusedResult.best.Score {
				fusedResult.best = result
			}
			fusedResult.score += 1.0 / (outerRRFK + float64(rank+1))
		}
	}

	if len(fused) == 0 {
		if opts.FallbackEnabled != nil && !*opts.FallbackEnabled {
			return &SearchResponse{
				RetrievalExhausted: exhausted,
				Status:             "success",
				Query:              query,
				Results:            []SearchResult{},
				SearchMethod:       "chunked_rrf_hybrid",
				FallbackTriggered:  false,
			}, nil
		}
		response, err := searchQuery(ctx, query, nil, opts)
		if response != nil {
			// The callback may return a cached response shared with other callers.
			copy := *response
			copy.RetrievalExhausted = exhausted && response.RetrievalExhausted
			response = &copy
		}
		return response, err
	}

	slices.SortFunc(fused, func(left, right fusedResult) int {
		if order := cmp.Compare(right.score, left.score); order != 0 {
			return order
		}
		return cmp.Compare(searchResultID(*left.best), searchResultID(*right.best))
	})
	if opts.Limit > 0 && len(fused) > opts.Limit {
		exhausted = false
		fused = fused[:opts.Limit]
	}

	response := &SearchResponse{
		RetrievalExhausted: exhausted,
		Status:             "success",
		Query:              query,
		Results:            make([]SearchResult, 0, len(fused)),
		TotalCandidates:    len(fusedIndexes),
		Returned:           len(fused),
		SearchMethod:       "chunked_rrf_hybrid",
		FallbackTriggered:  false,
	}
	for _, fusedResult := range fused {
		result := *fusedResult.best
		result.Score = fusedResult.score
		result.RRFScore = fusedResult.score
		result.VectorRank = 0
		result.BM25Rank = 0
		response.Results = append(response.Results, result)
	}
	return response, nil
}

func chunkCandidateLimit(limit int) int {
	candidateLimit := limit * 3
	if candidateLimit < minChunkCandidateLimit {
		candidateLimit = minChunkCandidateLimit
	}
	if candidateLimit > maxChunkCandidateLimit {
		candidateLimit = maxChunkCandidateLimit
	}
	return candidateLimit
}

func searchResultID(result SearchResult) string {
	if result.NodeID != "" {
		return string(result.NodeID)
	}
	return result.ID
}

func isFatalChunkedSearchError(predicate func(error) bool, err error) bool {
	return predicate != nil && predicate(err)
}
