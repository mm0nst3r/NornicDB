package search

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/orneryd/nornicdb/pkg/resultstream"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

func TestContinuationShortApproximateBatchDoesNotEndSearch(t *testing.T) {
	service := NewService(storage.NewMemoryEngine())
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	var depths []int
	search := func(_ context.Context, _ string, _ []float32, opts *SearchOptions) (*SearchResponse, error) {
		depths = append(depths, opts.Limit)
		count := 1
		if opts.Limit >= 8 {
			count = 3
		}
		results := make([]SearchResult, count)
		for i := range results {
			results[i] = SearchResult{ID: fmt.Sprintf("doc-%d", i)}
		}
		return &SearchResponse{Results: results}, nil
	}
	request := SearchContinuationRequest{Owner: "alice", N: 2, MaxResults: 3}
	page, err := service.SearchTextContinuation(context.Background(), "query", &SearchOptions{Limit: 4}, request, nil, nil, search, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.Len(t, page.Results, 2)
	require.True(t, page.HasMore)
	require.Contains(t, depths, 8)
	request.QID = page.QID
	page, err = service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.Len(t, page.Results, 1)
	require.Equal(t, "doc-2", page.Results[0].ID)
	require.False(t, page.HasMore)
}

func TestContinuationFilteredBM25EnumeratesBeyondShortFirstBatch(t *testing.T) {
	for _, engine := range []string{"v1", "v2"} {
		t.Run(engine, func(t *testing.T) {
			t.Setenv("NORNICDB_SEARCH_BM25_ENGINE", engine)
			testContinuationFilteredBM25(t)
		})
	}
}

func testContinuationFilteredBM25(t *testing.T) {
	ctx := context.Background()
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	for i := 0; i < 100; i++ {
		node := &storage.Node{ID: storage.NodeID(fmt.Sprintf("doc-%03d", i)), Properties: map[string]any{"content": "library searchable transcript"}}
		_, err := engine.CreateNode(node)
		require.NoError(t, err)
		require.NoError(t, service.IndexNode(node))
	}
	opts := DefaultSearchOptions()
	opts.Limit = 100
	all, err := service.Search(ctx, "transcript", nil, opts)
	require.NoError(t, err)
	require.Len(t, all.Results, 100)
	want := []string{searchResultID(all.Results[1]), searchResultID(all.Results[70])}
	for _, id := range want {
		node, err := engine.GetNode(storage.NodeID(id))
		require.NoError(t, err)
		node.Properties["eligible"] = "yes"
		require.NoError(t, engine.UpdateNode(node))
	}
	opts = DefaultSearchOptions()
	opts.Limit = 4
	opts.AdaptiveOverfetch = false
	opts.InitialOverfetchRatio = 1
	opts.Filters = map[string][]string{"eligible": {"yes"}}
	page, err := service.SearchTextContinuation(ctx, "transcript", opts, SearchContinuationRequest{Owner: "alice", N: 2}, nil, nil, service.Search, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	var got []string
	for _, result := range page.Results {
		got = append(got, searchResultID(result))
	}
	require.ElementsMatch(t, want, got)
	require.False(t, page.HasMore)
}

func TestContinuationUnknownExhaustionReachesExplicitBudget(t *testing.T) {
	service := NewService(storage.NewMemoryEngine())
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	var depths []int
	search := func(_ context.Context, _ string, _ []float32, opts *SearchOptions) (*SearchResponse, error) {
		depths = append(depths, opts.Limit)
		return &SearchResponse{Results: []SearchResult{{ID: "doc"}}}, nil
	}
	page, err := service.SearchTextContinuation(context.Background(), "query", &SearchOptions{Limit: 2, MaxCandidateLimit: 16}, SearchContinuationRequest{Owner: "alice", N: 2}, nil, nil, search, ChunkedSearchErrorPolicy{})
	require.ErrorIs(t, err, resultstream.ErrCapacity)
	require.Nil(t, page, "a resource limit must not return a successful exhausted page")
	require.Equal(t, []int{2, 4, 8, 16}, depths)
}

func TestContinuationShortExpansionDoesNotEndSearch(t *testing.T) {
	service := NewService(storage.NewMemoryEngine())
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	search := func(_ context.Context, _ string, _ []float32, opts *SearchOptions) (*SearchResponse, error) {
		count := 4
		if opts.Limit >= 8 {
			count = 5
		}
		if opts.Limit >= 16 {
			count = 7
		}
		results := make([]SearchResult, count)
		for i := range results {
			results[i] = SearchResult{ID: fmt.Sprintf("doc-%d", i)}
		}
		return &SearchResponse{Results: results}, nil
	}
	request := SearchContinuationRequest{Owner: "alice", N: 2, MaxResults: 7}
	page, err := service.SearchTextContinuation(context.Background(), "query", &SearchOptions{Limit: 4}, request, nil, nil, search, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	var ids []string
	for {
		for _, result := range page.Results {
			ids = append(ids, result.ID)
		}
		if !page.HasMore {
			break
		}
		request.QID = page.QID
		page, err = service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
		require.NoError(t, err)
		replay, err := service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
		require.NoError(t, err)
		require.Equal(t, page.Results, replay.Results)
	}
	require.Equal(t, []string{"doc-0", "doc-1", "doc-2", "doc-3", "doc-4", "doc-5", "doc-6"}, ids)
}

func TestContinuationVectorExhaustionRequiresExactCandidateEvidence(t *testing.T) {
	index := NewVectorIndex(2)
	hnsw := NewHNSWIndex(2, DefaultHNSWConfig())
	require.NoError(t, index.Add("a", []float32{1, 0}))
	require.NoError(t, hnsw.Add("a", []float32{1, 0}))
	for _, tc := range []struct {
		name      string
		generator CandidateGenerator
		exhausted bool
	}{
		{"exact", NewBruteForceCandidateGen(index), true},
		{"fully explored HNSW", NewHNSWCandidateGen(hnsw), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pipeline := NewVectorSearchPipeline(tc.generator, NewCPUExactScorer(index))
			results, exhausted, err := pipeline.searchWithExhaustion(context.Background(), []float32{1, 0}, 10, 0.5)
			require.NoError(t, err)
			require.Len(t, results, 1)
			require.Equal(t, tc.exhausted, exhausted)
		})
	}
}

func TestContinuationHNSWShortThresholdBatchNeedsFullCoverage(t *testing.T) {
	index := NewHNSWIndex(2, DefaultHNSWConfig())
	for i := 0; i < 64; i++ {
		angle := float64(i) / 64
		require.NoError(t, index.Add(fmt.Sprintf("v-%d", i), []float32{float32(math.Cos(angle)), float32(math.Sin(angle))}))
	}
	short, exhausted, err := index.searchWithEfExhaustion(context.Background(), []float32{1, 0}, 5, 0.99999, 5)
	require.NoError(t, err)
	require.Len(t, short, 1)
	require.False(t, exhausted, "a short result after threshold filtering is not corpus coverage")
	all, exhausted, err := index.searchWithEfExhaustion(context.Background(), []float32{1, 0}, 128, 0.99999, 128)
	require.NoError(t, err)
	require.Equal(t, short, all)
	require.True(t, exhausted, "the complete pre-filter heap supplies actual coverage evidence")
}

func TestContinuationDefaultHNSWFinishesFinitePopulation(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewServiceWithDimensions(engine, 2)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	for i := 0; i < 6; i++ {
		node := &storage.Node{ID: storage.NodeID(fmt.Sprintf("doc-%d", i)), ChunkEmbeddings: [][]float32{{1, float32(i) / 10}}}
		_, err := engine.CreateNode(node)
		require.NoError(t, err)
		require.NoError(t, service.IndexNode(node))
	}
	embedCalls := 0
	embed := func(context.Context, string) ([]float32, error) { embedCalls++; return []float32{1, 0}, nil }
	request := SearchContinuationRequest{Owner: "alice", N: 2}
	page, err := service.SearchTextContinuation(context.Background(), "", &SearchOptions{Limit: 2}, request, nil, embed, service.Search, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	ids := map[string]bool{}
	for {
		for _, result := range page.Results {
			id := searchResultID(result)
			require.False(t, ids[id], "duplicate %s", id)
			ids[id] = true
		}
		if !page.HasMore {
			break
		}
		request.QID = page.QID
		page, err = service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
		require.NoError(t, err)
	}
	require.Len(t, ids, 6)
	require.Equal(t, 1, embedCalls)
}

func TestContinuationChunkExhaustionSurvivesFusionAndFallback(t *testing.T) {
	chunk := func(context.Context, string) ([]string, error) { return []string{"one", "two"}, nil }
	embed := func(context.Context, string) ([]float32, error) { return []float32{1}, nil }
	for _, tc := range []struct {
		name  string
		known bool
		empty bool
	}{
		{"known", true, false}, {"unknown", false, false}, {"unknown with fallback", false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fallback := &SearchResponse{RetrievalExhausted: true}
			search := func(_ context.Context, _ string, vector []float32, _ *SearchOptions) (*SearchResponse, error) {
				if len(vector) == 0 {
					return fallback, nil
				}
				response := &SearchResponse{RetrievalExhausted: tc.known}
				if !tc.empty {
					response.Results = []SearchResult{{ID: "a"}}
				}
				return response, nil
			}
			response, err := SearchTextChunks(context.Background(), "query", &SearchOptions{Limit: 2}, chunk, embed, search)
			require.NoError(t, err)
			require.Equal(t, tc.known, response.RetrievalExhausted)
			require.True(t, fallback.RetrievalExhausted, "shared callback response must not be mutated")
		})
	}
}

func TestContinuationChunkDepthIsNotCappedAtOneShotLimit(t *testing.T) {
	service := NewService(storage.NewMemoryEngine())
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	embeds := 0
	search := func(_ context.Context, _ string, _ []float32, opts *SearchOptions) (*SearchResponse, error) {
		results := make([]SearchResult, min(opts.Limit, 600))
		for i := range results {
			results[i] = SearchResult{ID: fmt.Sprintf("doc-%04d", i)}
		}
		return &SearchResponse{Results: results}, nil
	}
	chunk := func(context.Context, string) ([]string, error) { return []string{"one", "two"}, nil }
	embed := func(context.Context, string) ([]float32, error) { embeds++; return []float32{1}, nil }
	request := SearchContinuationRequest{Owner: "alice", N: 200, MaxResults: 400}
	page, err := service.SearchTextContinuation(context.Background(), "query", &SearchOptions{Limit: 200}, request, chunk, embed, search, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.Len(t, page.Results, 200)
	require.True(t, page.HasMore)
	request.QID = page.QID
	page, err = service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.Len(t, page.Results, 200)
	require.False(t, page.HasMore)
	require.Equal(t, 2, embeds)
}
