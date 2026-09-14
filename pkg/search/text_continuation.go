package search

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/orneryd/nornicdb/pkg/resultstream"
)

// SearchContinuationRequest controls one start, pull, or discard operation.
// Limit belongs to SearchOptions and is only the initial retrieval depth.
type SearchContinuationRequest struct {
	Owner      string
	Database   string
	QID        string
	N          int
	Discard    bool
	MaxResults int
}

// SearchContinuationPage is the protocol-neutral continued-search response.
type SearchContinuationPage struct {
	Results           []SearchResult
	QID               string
	HasMore           bool
	Position          uint64
	Returned          int
	Discovered        int
	Total             *uint64
	ExpiresAt         time.Time
	Released          bool
	SearchMethod      string
	FallbackTriggered bool
}

type continuationRegistry interface {
	Start(context.Context, resultstream.Scope, resultstream.Stream, int) (*resultstream.Page, error)
	Pull(context.Context, resultstream.Scope, string, int) (*resultstream.Page, error)
	Discard(resultstream.Scope, string) error
	Close()
}

type cachedTextPreparation struct {
	chunkOnce sync.Once
	chunks    []string
	chunkErr  error

	embedMu sync.Mutex
	embeds  map[string]cachedEmbedding
}

type cachedEmbedding struct {
	vector []float32
	err    error
}

type searchContinuationState struct {
	mu                sync.RWMutex
	results           []SearchResult
	seen              map[string]struct{}
	searchMethod      string
	fallbackTriggered bool
}

// SearchTextContinuation starts or resumes a progressively deepened canonical
// text search. Chunking and embedding execute only while starting the stream.
func (s *Service) SearchTextContinuation(
	ctx context.Context,
	query string,
	opts *SearchOptions,
	request SearchContinuationRequest,
	chunkQuery ChunkQueryFunc,
	embedQuery EmbedQueryFunc,
	searchQuery SearchQueryFunc,
	errorPolicy ChunkedSearchErrorPolicy,
) (*SearchContinuationPage, error) {
	registry, err := s.searchContinuationRegistry()
	if err != nil {
		return nil, err
	}
	database := request.Database
	if database == "" {
		database = s.cacheNamespace
	}
	if database == "" {
		database = "default"
	}
	scope := resultstream.Scope{Owner: request.Owner, Database: database}
	if request.Discard {
		if request.QID == "" {
			return nil, resultstream.ErrInvalidQID
		}
		if err := registry.Discard(scope, request.QID); err != nil {
			return nil, err
		}
		return &SearchContinuationPage{Released: true}, nil
	}
	if request.QID != "" {
		page, err := registry.Pull(ctx, scope, request.QID, request.N)
		if err != nil {
			return nil, err
		}
		return searchPageFromResultStream(page)
	}
	if searchQuery == nil {
		return nil, errors.New("search continuation requires a search function")
	}
	if opts == nil {
		opts = DefaultSearchOptions()
	}
	ownedOptions := cloneContinuationSearchOptions(opts)
	if ownedOptions.Limit <= 0 {
		ownedOptions.Limit = 50
	}
	if ownedOptions.Limit < request.N {
		ownedOptions.Limit = request.N
	}
	preparation := &cachedTextPreparation{embeds: make(map[string]cachedEmbedding)}
	cachedChunks := preparation.chunker(chunkQuery)
	cachedEmbeds := preparation.embedder(embedQuery)
	initial, err := SearchTextChunksWithErrorPolicy(ctx, query, &ownedOptions, cachedChunks, cachedEmbeds, searchQuery, errorPolicy)
	if err != nil {
		return nil, err
	}
	state := &searchContinuationState{
		results:           append([]SearchResult(nil), initial.Results...),
		seen:              make(map[string]struct{}, len(initial.Results)),
		searchMethod:      initial.SearchMethod,
		fallbackTriggered: initial.FallbackTriggered,
	}
	for index := range state.results {
		state.seen[searchResultID(state.results[index])] = struct{}{}
	}
	initialRows := continuationRows(state.results)
	maxResults := request.MaxResults
	initialExhausted := len(initial.Results) < ownedOptions.Limit ||
		(maxResults > 0 && len(initial.Results) >= maxResults)
	expand := func(expandCtx context.Context, depth int) ([][]any, bool, error) {
		if maxResults > 0 && depth > maxResults {
			depth = maxResults
		}
		expandedOptions := cloneContinuationSearchOptions(&ownedOptions)
		expandedOptions.Limit = depth
		response, expandErr := SearchTextChunksWithErrorPolicy(expandCtx, query, &expandedOptions, cachedChunks, cachedEmbeds, searchQuery, errorPolicy)
		if expandErr != nil {
			return nil, false, expandErr
		}
		state.mu.Lock()
		defer state.mu.Unlock()
		for index := range response.Results {
			result := response.Results[index]
			id := searchResultID(result)
			if _, exists := state.seen[id]; exists {
				continue
			}
			state.seen[id] = struct{}{}
			state.results = append(state.results, result)
		}
		exhausted := len(response.Results) < depth ||
			(maxResults > 0 && len(state.results) >= maxResults)
		if maxResults > 0 && len(state.results) > maxResults {
			state.results = state.results[:maxResults]
		}
		return continuationRows(state.results), exhausted, nil
	}
	stream, err := resultstream.NewProgressive(initialRows, initialExhausted, ownedOptions.Limit, expand)
	if err != nil {
		return nil, err
	}
	page, err := registry.Start(ctx, scope, &searchMetadataStream{Stream: stream, state: state}, request.N)
	if err != nil {
		return nil, err
	}
	return searchPageFromResultStream(page)
}

func (s *Service) searchContinuationRegistry() (continuationRegistry, error) {
	s.continuationMu.Lock()
	defer s.continuationMu.Unlock()
	if s.continuationRegistry != nil {
		return s.continuationRegistry, nil
	}
	registry, err := resultstream.NewRegistry(resultstream.Config{})
	if err != nil {
		return nil, err
	}
	s.continuationRegistry = registry
	s.continuationOwned = true
	return registry, nil
}

// SetContinuationRegistry shares one process registry across database-scoped
// search services. The registry remains owned by the caller.
func (s *Service) SetContinuationRegistry(registry *resultstream.Registry) {
	if s == nil || registry == nil {
		return
	}
	s.continuationMu.Lock()
	if s.continuationRegistry != nil && s.continuationOwned {
		s.continuationRegistry.Close()
	}
	s.continuationRegistry = registry
	s.continuationOwned = false
	s.continuationMu.Unlock()
}

func (p *cachedTextPreparation) chunker(chunkQuery ChunkQueryFunc) ChunkQueryFunc {
	if chunkQuery == nil {
		return nil
	}
	return func(ctx context.Context, query string) ([]string, error) {
		p.chunkOnce.Do(func() {
			p.chunks, p.chunkErr = chunkQuery(ctx, query)
			p.chunks = append([]string(nil), p.chunks...)
		})
		return p.chunks, p.chunkErr
	}
}

func (p *cachedTextPreparation) embedder(embedQuery EmbedQueryFunc) EmbedQueryFunc {
	if embedQuery == nil {
		return nil
	}
	return func(ctx context.Context, query string) ([]float32, error) {
		p.embedMu.Lock()
		cached, exists := p.embeds[query]
		p.embedMu.Unlock()
		if exists {
			return cached.vector, cached.err
		}
		vector, err := embedQuery(ctx, query)
		vector = append([]float32(nil), vector...)
		p.embedMu.Lock()
		p.embeds[query] = cachedEmbedding{vector: vector, err: err}
		p.embedMu.Unlock()
		return vector, err
	}
}

type searchMetadataStream struct {
	resultstream.Stream
	state *searchContinuationState
}

func (s *searchMetadataStream) Pull(ctx context.Context, position uint64, n int) (*resultstream.Page, error) {
	page, err := s.Stream.Pull(ctx, position, n)
	if err != nil {
		return nil, err
	}
	s.state.mu.RLock()
	page.Metadata = map[string]any{
		"search_method":      s.state.searchMethod,
		"fallback_triggered": s.state.fallbackTriggered,
		"discovered":         len(s.state.results),
	}
	s.state.mu.RUnlock()
	return page, nil
}

func continuationRows(results []SearchResult) [][]any {
	rows := make([][]any, len(results))
	for index := range results {
		rows[index] = []any{results[index]}
	}
	return rows
}

func searchPageFromResultStream(page *resultstream.Page) (*SearchContinuationPage, error) {
	results := make([]SearchResult, len(page.Rows))
	for index := range page.Rows {
		if len(page.Rows[index]) != 1 {
			return nil, resultstream.ErrInvalidPosition
		}
		result, ok := page.Rows[index][0].(SearchResult)
		if !ok {
			return nil, resultstream.ErrInvalidPosition
		}
		results[index] = result
	}
	out := &SearchContinuationPage{
		Results:   results,
		QID:       page.QID,
		HasMore:   page.HasMore,
		Position:  page.Position,
		Returned:  len(results),
		Total:     page.Total,
		ExpiresAt: page.ExpiresAt,
	}
	if page.Metadata != nil {
		out.SearchMethod, _ = page.Metadata["search_method"].(string)
		out.FallbackTriggered, _ = page.Metadata["fallback_triggered"].(bool)
		out.Discovered, _ = page.Metadata["discovered"].(int)
	}
	return out, nil
}

func cloneContinuationSearchOptions(options *SearchOptions) SearchOptions {
	clone := *options
	clone.Types = append([]string(nil), options.Types...)
	if options.MinSimilarity != nil {
		value := *options.MinSimilarity
		clone.MinSimilarity = &value
	}
	if options.FallbackEnabled != nil {
		value := *options.FallbackEnabled
		clone.FallbackEnabled = &value
	}
	if options.Filters != nil {
		clone.Filters = make(map[string][]string, len(options.Filters))
		for key, values := range options.Filters {
			clone.Filters[key] = append([]string(nil), values...)
		}
	}
	return clone
}
