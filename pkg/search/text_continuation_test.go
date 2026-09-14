package search

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/orneryd/nornicdb/pkg/resultstream"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

type blockingContinuationEngine struct {
	storage.Engine
	streamer storage.StreamingEngine
	started  chan struct{}
	resume   chan struct{}
	once     sync.Once
}

type nonStreamingContinuationEngine struct {
	storage.Engine
}

func (e nonStreamingContinuationEngine) GraphMutationVersion() (uint64, bool) {
	return e.Engine.(storage.GraphMutationVersionProvider).GraphMutationVersion()
}

func (e *blockingContinuationEngine) GraphMutationVersion() (uint64, bool) {
	return e.Engine.(storage.GraphMutationVersionProvider).GraphMutationVersion()
}

func (e *blockingContinuationEngine) StreamNodes(ctx context.Context, fn func(*storage.Node) error) error {
	return e.streamer.StreamNodes(ctx, func(node *storage.Node) error {
		e.once.Do(func() {
			close(e.started)
			<-e.resume
		})
		return fn(node)
	})
}

func (e *blockingContinuationEngine) StreamEdges(ctx context.Context, fn func(*storage.Edge) error) error {
	return e.streamer.StreamEdges(ctx, fn)
}

func (e *blockingContinuationEngine) StreamNodeChunks(ctx context.Context, chunkSize int, fn func([]*storage.Node) error) error {
	return e.streamer.StreamNodeChunks(ctx, chunkSize, fn)
}

func TestSearchTextContinuationExpandsWithoutReembedding(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	for index := 0; index < 30; index++ {
		id := storage.NodeID(fmt.Sprintf("node-%03d", index))
		_, err := engine.CreateNode(&storage.Node{ID: id, Labels: []string{"Document"}})
		require.NoError(t, err)
	}

	var chunkCalls atomic.Int32
	var embedCalls atomic.Int32
	chunkQuery := func(context.Context, string) ([]string, error) {
		chunkCalls.Add(1)
		return []string{"one", "two"}, nil
	}
	embedQuery := func(_ context.Context, query string) ([]float32, error) {
		embedCalls.Add(1)
		if query == "one" {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	}
	searchQuery := func(_ context.Context, _ string, _ []float32, opts *SearchOptions) (*SearchResponse, error) {
		results := make([]SearchResult, opts.Limit)
		for index := range results {
			id := fmt.Sprintf("node-%03d", index)
			results[index] = SearchResult{ID: id, NodeID: storage.NodeID(id), Score: float64(opts.Limit - index)}
		}
		return &SearchResponse{Status: "success", Results: results, SearchMethod: "test"}, nil
	}

	request := SearchContinuationRequest{
		Owner:    "alice",
		Database: "nornic",
		N:        2,
	}
	options := &SearchOptions{Limit: 2}
	first, err := service.SearchTextContinuation(context.Background(), "query", options, request, chunkQuery, embedQuery, searchQuery, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.Len(t, first.Results, 2)
	require.True(t, first.HasMore)
	require.NotEmpty(t, first.QID)

	request.QID = first.QID
	request.N = 4
	second, err := service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.Len(t, second.Results, 4)
	require.Equal(t, "node-002", second.Results[0].ID)
	seen := map[string]struct{}{}
	for _, page := range [][]SearchResult{first.Results, second.Results} {
		for _, result := range page {
			_, duplicate := seen[result.ID]
			require.False(t, duplicate, "duplicate result %s", result.ID)
			seen[result.ID] = struct{}{}
		}
	}
	require.Len(t, seen, 6)
	require.Equal(t, int32(1), chunkCalls.Load())
	require.Equal(t, int32(2), embedCalls.Load())
}

func TestSearchTextContinuationMaxResultsCapsInitialPage(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	for index := 0; index < 10; index++ {
		id := storage.NodeID(fmt.Sprintf("node-%03d", index))
		_, err := engine.CreateNode(&storage.Node{ID: id, Labels: []string{"Document"}})
		require.NoError(t, err)
	}

	searchQuery := func(_ context.Context, _ string, _ []float32, opts *SearchOptions) (*SearchResponse, error) {
		results := make([]SearchResult, opts.Limit)
		for index := range results {
			id := fmt.Sprintf("node-%03d", index)
			results[index] = SearchResult{ID: id, NodeID: storage.NodeID(id)}
		}
		return &SearchResponse{Status: "success", Results: results, SearchMethod: "test"}, nil
	}

	page, err := service.SearchTextContinuation(
		context.Background(),
		"query",
		&SearchOptions{Limit: 10},
		SearchContinuationRequest{Owner: "alice", Database: "nornic", N: 10, MaxResults: 3},
		nil,
		nil,
		searchQuery,
		ChunkedSearchErrorPolicy{},
	)
	require.NoError(t, err)
	require.Len(t, page.Results, 3)
	require.False(t, page.HasMore)
	require.Empty(t, page.QID)
}

func TestSearchTextContinuationIDModeGroupsCompleteFilteredPopulation(t *testing.T) {
	base := storage.NewMemoryEngine()
	engine := storage.NewNamespacedEngine(base, "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })

	for _, node := range []*storage.Node{
		{ID: "frame-c", Labels: []string{"Frame"}, Properties: map[string]any{"collection": "summer", "asset_id": "asset-b"}},
		{ID: "frame-b", Labels: []string{"Frame"}, Properties: map[string]any{"collection": "summer", "asset_id": "asset-a"}},
		{ID: "frame-a", Labels: []string{"Frame"}, Properties: map[string]any{"collection": "summer", "asset_id": "asset-a"}},
		{ID: "winter", Labels: []string{"Frame"}, Properties: map[string]any{"collection": "winter", "asset_id": "asset-c"}},
	} {
		_, err := engine.CreateNode(node)
		require.NoError(t, err)
	}

	request := SearchContinuationRequest{
		Owner:    "alice",
		Database: "nornic",
		Mode:     SearchContinuationID,
		GroupBy:  "asset_id",
		N:        1,
	}
	options := &SearchOptions{
		Types:   []string{"frame"},
		Filters: map[string][]string{"collection": {"summer"}},
	}
	first, err := service.SearchTextContinuation(context.Background(), "", options, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.Len(t, first.Results, 1)
	require.Equal(t, "frame-a", first.Results[0].ID)
	require.Equal(t, "asset-a", first.Results[0].GroupKey)
	require.Equal(t, []string{"frame-a", "frame-b"}, passageIDs(first.Results[0].Passages))
	require.Equal(t, SearchContinuationCatalogPhase, first.Results[0].Phase)
	require.Equal(t, SearchContinuationID, first.Mode)
	require.Equal(t, 2, *first.EligibleCount)
	require.True(t, first.RankedPoolExhausted)
	require.False(t, first.CollectionExhausted)
	require.NotEmpty(t, first.QID)

	request.QID = first.QID
	second, err := service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.Len(t, second.Results, 1)
	require.Equal(t, "frame-c", second.Results[0].ID)
	require.Equal(t, "asset-b", second.Results[0].GroupKey)
	require.True(t, second.CollectionExhausted)
	require.Equal(t, SearchContinuationCollectionComplete, second.Completion)
	require.Empty(t, second.QID)
}

func TestSearchTextContinuationRankedThenIDFreezesPrefixBeforeCatalog(t *testing.T) {
	base := storage.NewMemoryEngine()
	engine := storage.NewNamespacedEngine(base, "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })

	for _, node := range []*storage.Node{
		{ID: "frame-a", Labels: []string{"Frame"}, Properties: map[string]any{"collection": "summer", "asset_id": "asset-a"}},
		{ID: "frame-b", Labels: []string{"Frame"}, Properties: map[string]any{"collection": "summer", "asset_id": "asset-a"}},
		{ID: "frame-c", Labels: []string{"Frame"}, Properties: map[string]any{"collection": "summer", "asset_id": "asset-b"}},
		{ID: "frame-d", Labels: []string{"Frame"}, Properties: map[string]any{"collection": "summer", "asset_id": "asset-c"}},
	} {
		_, err := engine.CreateNode(node)
		require.NoError(t, err)
	}

	var searchCalls atomic.Int32
	searchQuery := func(_ context.Context, _ string, _ []float32, opts *SearchOptions) (*SearchResponse, error) {
		searchCalls.Add(1)
		results := []SearchResult{
			{ID: "frame-b", NodeID: "frame-b", Score: 0.9},
			{ID: "frame-a", NodeID: "frame-a", Score: 0.85},
			{ID: "frame-c", NodeID: "frame-c", Score: 0.8},
		}
		return &SearchResponse{Results: results[:min(opts.Limit, len(results))], SearchMethod: "test"}, nil
	}
	request := SearchContinuationRequest{
		Owner: "alice", Database: "nornic", Mode: SearchContinuationRankedThenID,
		GroupBy: "asset_id", RankedLimit: 3, N: 2,
	}
	options := &SearchOptions{Limit: 1, Types: []string{"frame"}, Filters: map[string][]string{"collection": {"summer"}}}
	first, err := service.SearchTextContinuation(
		context.Background(), "query", options, request,
		func(context.Context, string) ([]string, error) { return []string{"query"}, nil },
		func(context.Context, string) ([]float32, error) { return []float32{1}, nil },
		searchQuery, ChunkedSearchErrorPolicy{},
	)
	require.NoError(t, err)
	require.Equal(t, []string{"frame-b", "frame-c"}, []string{first.Results[0].ID, first.Results[1].ID})
	require.Equal(t, []string{"frame-b", "frame-a"}, passageIDs(first.Results[0].Passages))
	require.Equal(t, []string{SearchContinuationRankedPhase, SearchContinuationRankedPhase}, []string{first.Results[0].Phase, first.Results[1].Phase})
	require.Equal(t, 2, first.RankedCount)
	require.False(t, first.RankedPoolExhausted)
	require.False(t, first.CollectionExhausted)

	request.QID = first.QID
	last, err := service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.Len(t, last.Results, 1)
	require.Equal(t, "frame-d", last.Results[0].ID)
	require.Equal(t, "asset-c", last.Results[0].GroupKey)
	require.Equal(t, SearchContinuationCatalogPhase, last.Results[0].Phase)
	require.True(t, last.CollectionExhausted)
	require.Equal(t, int32(1), searchCalls.Load())
}

func TestSearchTextContinuationRankedThenIDUsesBranchLimitWhenRankedLimitOmitted(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	_, err := engine.CreateNode(&storage.Node{ID: "doc-a", Labels: []string{"Document"}, Properties: map[string]any{"content": "alpha"}})
	require.NoError(t, err)

	requestedLimits := []int{}
	searchQuery := func(_ context.Context, _ string, _ []float32, opts *SearchOptions) (*SearchResponse, error) {
		requestedLimits = append(requestedLimits, opts.Limit)
		return &SearchResponse{
			Results:      []SearchResult{{ID: "doc-a", NodeID: "doc-a", Score: 1}},
			SearchMethod: "test", RetrievalExhausted: opts.Limit >= 2,
		}, nil
	}
	options := DefaultSearchOptions()
	options.Limit = 1
	page, err := service.SearchTextContinuation(
		context.Background(), "alpha", options,
		SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationRankedThenID, N: 1},
		nil, nil, searchQuery, ChunkedSearchErrorPolicy{},
	)
	require.NoError(t, err)
	require.Equal(t, []int{1, 2}, requestedLimits)
	require.Len(t, page.Results, 1)
}

func TestSearchTextContinuationRankedPullRehydratesAndAuthorizes(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	for _, id := range []storage.NodeID{"doc-a", "doc-b"} {
		_, err := engine.CreateNode(&storage.Node{ID: id, Labels: []string{"Document"}, Properties: map[string]any{"version": "current"}})
		require.NoError(t, err)
	}
	var allowSecond atomic.Bool
	allowSecond.Store(true)
	searchQuery := func(context.Context, string, []float32, *SearchOptions) (*SearchResponse, error) {
		return &SearchResponse{Results: []SearchResult{
			{ID: "doc-a", NodeID: "doc-a", Score: 1, Properties: map[string]any{"version": "stale"}},
			{ID: "doc-b", NodeID: "doc-b", Score: .9, Properties: map[string]any{"version": "stale"}},
		}, RetrievalExhausted: true}, nil
	}
	request := SearchContinuationRequest{
		Owner: "alice", Database: "nornic", N: 1,
		AuthorizeNode: func(node *storage.Node) (bool, error) { return node.ID != "doc-b" || allowSecond.Load(), nil },
	}
	first, err := service.SearchTextContinuation(context.Background(), "query", &SearchOptions{Limit: 2}, request, nil, nil, searchQuery, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.Equal(t, "current", first.Results[0].Properties["version"])

	allowSecond.Store(false)
	request.QID = first.QID
	_, err = service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.ErrorIs(t, err, resultstream.ErrInvalidated)
}

func TestSearchTextContinuationIDModeMaxResultsReturnsSortedPrefix(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	for _, id := range []storage.NodeID{"doc-c", "doc-a", "doc-b"} {
		_, err := engine.CreateNode(&storage.Node{ID: id, Labels: []string{"Document"}})
		require.NoError(t, err)
	}
	page, err := service.SearchTextContinuation(
		context.Background(), "", DefaultSearchOptions(),
		SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, N: 2, MaxResults: 2},
		nil, nil, nil, ChunkedSearchErrorPolicy{},
	)
	require.NoError(t, err)
	require.Equal(t, []string{"doc-a", "doc-b"}, []string{page.Results[0].ID, page.Results[1].ID})
	require.Equal(t, 3, *page.EligibleCount)
	require.False(t, page.HasMore)
}

func TestSearchTextContinuationPreservesCanonicalResponseMetadata(t *testing.T) {
	for _, mode := range []SearchContinuationMode{SearchContinuationRanked, SearchContinuationRankedThenID} {
		t.Run(string(mode), func(t *testing.T) {
			engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
			service := NewService(engine)
			t.Cleanup(func() { require.NoError(t, service.Close()) })
			for _, id := range []storage.NodeID{"a", "b"} {
				_, err := engine.CreateNode(&storage.Node{ID: id, Properties: map[string]any{"content": "query"}})
				require.NoError(t, err)
			}
			retrieve := func(context.Context, string, []float32, *SearchOptions) (*SearchResponse, error) {
				return &SearchResponse{Results: []SearchResult{{ID: "a", NodeID: "a"}, {ID: "b", NodeID: "b"}}, Message: "provider outcome", RetrievalExhausted: true}, nil
			}
			request := SearchContinuationRequest{Owner: "alice", Mode: mode, N: 1, RankedLimit: 2}
			page, err := service.SearchTextContinuation(context.Background(), "query", DefaultSearchOptions(), request, nil, nil, retrieve, ChunkedSearchErrorPolicy{})
			require.NoError(t, err)
			require.Equal(t, "provider outcome", page.SearchResponse().Message)
			require.Len(t, page.SearchResponse().Results, 1)
			request.QID = page.QID
			page, err = service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
			require.NoError(t, err)
			require.Equal(t, "provider outcome", page.SearchResponse().Message)
			require.Len(t, page.SearchResponse().Results, 1)
		})
	}
}

func TestSearchTextContinuationIDModeCountsRejectedNodesAgainstScanLimit(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	service.SetCompleteContinuationPolicy(CompleteContinuationPolicy{MaxScannedNodes: 2})
	t.Cleanup(func() { require.NoError(t, service.Close()) })

	for _, node := range []*storage.Node{
		{ID: "winter-a", Labels: []string{"Frame"}, Properties: map[string]any{"collection": "winter"}},
		{ID: "winter-b", Labels: []string{"Frame"}, Properties: map[string]any{"collection": "winter"}},
		{ID: "summer", Labels: []string{"Frame"}, Properties: map[string]any{"collection": "summer"}},
	} {
		_, err := engine.CreateNode(node)
		require.NoError(t, err)
	}

	_, err := service.SearchTextContinuation(
		context.Background(), "",
		&SearchOptions{Filters: map[string][]string{"collection": {"summer"}}},
		SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, N: 1},
		nil, nil, nil, ChunkedSearchErrorPolicy{},
	)
	require.ErrorIs(t, err, resultstream.ErrCapacity)
}

func TestSearchTextContinuationIDModeFailsInsteadOfTruncatingLogicalMembers(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	service.SetCompleteContinuationPolicy(CompleteContinuationPolicy{MaxMembers: 1})
	t.Cleanup(func() { require.NoError(t, service.Close()) })

	for _, node := range []*storage.Node{
		{ID: "frame-a", Labels: []string{"Frame"}, Properties: map[string]any{"asset_id": "asset-a"}},
		{ID: "frame-b", Labels: []string{"Frame"}, Properties: map[string]any{"asset_id": "asset-b"}},
	} {
		_, err := engine.CreateNode(node)
		require.NoError(t, err)
	}

	_, err := service.SearchTextContinuation(
		context.Background(), "", DefaultSearchOptions(),
		SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, GroupBy: "asset_id", N: 1},
		nil, nil, nil, ChunkedSearchErrorPolicy{},
	)
	require.True(t, errors.Is(err, resultstream.ErrCapacity), "expected capacity error, got %v", err)
}

func TestSearchTextContinuationIDModeBoundsGroupedPassageFanout(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	service.SetCompleteContinuationPolicy(CompleteContinuationPolicy{MaxMembers: 1, MaxPassages: 2})
	t.Cleanup(func() { require.NoError(t, service.Close()) })

	for _, id := range []string{"frame-a", "frame-b", "frame-c"} {
		_, err := engine.CreateNode(&storage.Node{
			ID: storage.NodeID(id), Labels: []string{"Frame"},
			Properties: map[string]any{"asset_id": "asset-a"},
		})
		require.NoError(t, err)
	}

	_, err := service.SearchTextContinuation(
		context.Background(), "", DefaultSearchOptions(),
		SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, GroupBy: "asset_id", N: 1},
		nil, nil, nil, ChunkedSearchErrorPolicy{},
	)
	require.ErrorIs(t, err, resultstream.ErrCapacity)
}

func TestSearchTextContinuationIDModeRejectsBuildWhenAdmissionIsFull(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	service.SetCompleteContinuationPolicy(CompleteContinuationPolicy{MaxConcurrentBuilds: 1})
	t.Cleanup(func() { require.NoError(t, service.Close()) })

	_, _, admitted := service.acquireCompleteContinuationBuild()
	require.True(t, admitted)
	t.Cleanup(func() { service.completeBuilds.Add(-1) })

	_, err := service.SearchTextContinuation(
		context.Background(), "", DefaultSearchOptions(),
		SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, N: 1},
		nil, nil, nil, ChunkedSearchErrorPolicy{},
	)
	require.ErrorIs(t, err, resultstream.ErrCapacity)
}

func TestSearchTextContinuationIDModeBoundsRetainedDescriptorBytes(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	service.SetCompleteContinuationPolicy(CompleteContinuationPolicy{MaxBuildBytes: 1})
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	_, err := engine.CreateNode(&storage.Node{ID: "node-a", Labels: []string{"Document"}})
	require.NoError(t, err)

	_, err = service.SearchTextContinuation(
		context.Background(), "", DefaultSearchOptions(),
		SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, N: 1},
		nil, nil, nil, ChunkedSearchErrorPolicy{},
	)
	require.ErrorIs(t, err, resultstream.ErrCapacity)
}

func TestSearchTextContinuationIDModeBoundsBuildTime(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	service.SetCompleteContinuationPolicy(CompleteContinuationPolicy{MaxBuildDuration: time.Nanosecond})
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	_, err := engine.CreateNode(&storage.Node{ID: "node-a", Labels: []string{"Document"}})
	require.NoError(t, err)

	_, err = service.SearchTextContinuation(
		context.Background(), "", DefaultSearchOptions(),
		SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, N: 1},
		nil, nil, nil, ChunkedSearchErrorPolicy{},
	)
	require.ErrorIs(t, err, resultstream.ErrCapacity)
}

func TestSearchTextContinuationIDModeGroupingIsIndependentOfScanOrder(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })

	for _, idAndAsset := range [][2]string{
		{"frame-z", "asset-b"},
		{"frame-c", "asset-a"},
		{"frame-y", "asset-b"},
		{"frame-a", "asset-a"},
		{"frame-x", "asset-b"},
		{"frame-b", "asset-a"},
	} {
		_, err := engine.CreateNode(&storage.Node{
			ID: storage.NodeID(idAndAsset[0]), Labels: []string{"Frame"},
			Properties: map[string]any{"asset_id": idAndAsset[1]},
		})
		require.NoError(t, err)
	}

	page, err := service.SearchTextContinuation(
		context.Background(), "", DefaultSearchOptions(),
		SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, GroupBy: "asset_id", N: 10},
		nil, nil, nil, ChunkedSearchErrorPolicy{},
	)
	require.NoError(t, err)
	require.Equal(t, []string{"asset-a", "asset-b"}, []string{page.Results[0].GroupKey, page.Results[1].GroupKey})
	require.Equal(t, []string{"frame-a", "frame-b", "frame-c"}, passageIDs(page.Results[0].Passages))
	require.Equal(t, []string{"frame-x", "frame-y", "frame-z"}, passageIDs(page.Results[1].Passages))
}

func TestSearchTextContinuationIDModeRejectsInvalidGroupKeys(t *testing.T) {
	testCases := []struct {
		name       string
		properties map[string]any
	}{
		{name: "missing", properties: map[string]any{}},
		{name: "empty", properties: map[string]any{"asset_id": ""}},
		{name: "non-string", properties: map[string]any{"asset_id": 42}},
		{name: "invalid UTF-8", properties: map[string]any{"asset_id": string([]byte{0xff})}},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
			service := NewService(engine)
			t.Cleanup(func() { require.NoError(t, service.Close()) })
			_, err := engine.CreateNode(&storage.Node{ID: "frame-a", Labels: []string{"Frame"}, Properties: testCase.properties})
			require.NoError(t, err)

			_, err = service.SearchTextContinuation(
				context.Background(), "", DefaultSearchOptions(),
				SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, GroupBy: "asset_id", N: 1},
				nil, nil, nil, ChunkedSearchErrorPolicy{},
			)
			require.ErrorContains(t, err, `group property "asset_id" must be a nonempty UTF-8 string`)
		})
	}
}

func TestSearchTextContinuationIDModeInvalidatesAfterMutation(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	for _, id := range []string{"node-a", "node-b"} {
		_, err := engine.CreateNode(&storage.Node{ID: storage.NodeID(id), Labels: []string{"Document"}})
		require.NoError(t, err)
	}

	request := SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, N: 1}
	first, err := service.SearchTextContinuation(context.Background(), "", DefaultSearchOptions(), request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.NotEmpty(t, first.QID)
	_, err = engine.CreateNode(&storage.Node{ID: "node-c", Labels: []string{"Document"}})
	require.NoError(t, err)

	request.QID = first.QID
	_, err = service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.ErrorIs(t, err, resultstream.ErrInvalidated)
}

func TestSearchTextContinuationIDModeRejectsMutationDuringBuild(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	_, err := engine.CreateNode(&storage.Node{ID: "node-a", Labels: []string{"Document"}})
	require.NoError(t, err)
	blocking := &blockingContinuationEngine{
		Engine: engine, streamer: engine,
		started: make(chan struct{}), resume: make(chan struct{}),
	}
	service := NewService(blocking)
	t.Cleanup(func() { require.NoError(t, service.Close()) })

	result := make(chan error, 1)
	go func() {
		_, searchErr := service.SearchTextContinuation(
			context.Background(), "", DefaultSearchOptions(),
			SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, N: 1},
			nil, nil, nil, ChunkedSearchErrorPolicy{},
		)
		result <- searchErr
	}()
	<-blocking.started
	_, err = engine.CreateNode(&storage.Node{ID: "node-b", Labels: []string{"Document"}})
	require.NoError(t, err)
	close(blocking.resume)
	require.ErrorIs(t, <-result, resultstream.ErrInvalidated)
}

func TestSearchTextContinuationIDModeInvalidatesAfterPolicyChange(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	for _, id := range []string{"node-a", "node-b"} {
		_, err := engine.CreateNode(&storage.Node{ID: storage.NodeID(id), Labels: []string{"Document"}})
		require.NoError(t, err)
	}

	request := SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, N: 1}
	first, err := service.SearchTextContinuation(context.Background(), "", DefaultSearchOptions(), request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.NotEmpty(t, first.QID)
	service.SetCompleteContinuationPolicy(CompleteContinuationPolicy{MaxMembers: 10})

	request.QID = first.QID
	_, err = service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.ErrorIs(t, err, resultstream.ErrInvalidated)
}

func TestSearchTextContinuationIDModeAuthorizesBuildAndHydration(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	for _, id := range []string{"allowed-a", "allowed-b", "denied"} {
		_, err := engine.CreateNode(&storage.Node{ID: storage.NodeID(id), Labels: []string{"Document"}})
		require.NoError(t, err)
	}

	var allowSecond atomic.Bool
	allowSecond.Store(true)
	authorize := func(node *storage.Node) (bool, error) {
		if node.ID == "denied" {
			return false, nil
		}
		return node.ID != "allowed-b" || allowSecond.Load(), nil
	}
	request := SearchContinuationRequest{
		Owner: "alice", Database: "nornic", Mode: SearchContinuationID, N: 1,
		AuthorizeNode: authorize,
	}
	first, err := service.SearchTextContinuation(context.Background(), "", DefaultSearchOptions(), request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.NoError(t, err)
	require.Equal(t, "allowed-a", first.Results[0].ID)
	require.Equal(t, 2, *first.EligibleCount)
	require.True(t, first.HasMore)

	allowSecond.Store(false)
	request.QID = first.QID
	_, err = service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{})
	require.ErrorIs(t, err, resultstream.ErrInvalidated)
}

func TestCatalogContinuationCloseDoesNotWaitForHydration(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	for _, id := range []storage.NodeID{"doc-a", "doc-b"} {
		_, err := engine.CreateNode(&storage.Node{ID: id, Labels: []string{"Document"}})
		require.NoError(t, err)
	}
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	stream, err := service.newIDContinuationStream(context.Background(), *DefaultSearchOptions(), SearchContinuationRequest{
		Mode: SearchContinuationID,
	})
	require.NoError(t, err)
	catalog := stream.(*catalogContinuationStream)
	started := make(chan struct{})
	release := make(chan struct{})
	catalog.authorizeNode = func(*storage.Node) (bool, error) {
		select {
		case <-started:
		default:
			close(started)
		}
		<-release
		return true, nil
	}
	pullDone := make(chan error, 1)
	go func() {
		_, pullErr := catalog.Pull(context.Background(), 0, 1)
		pullDone <- pullErr
	}()
	<-started
	closeDone := make(chan error, 1)
	go func() { closeDone <- catalog.Close() }()
	select {
	case closeErr := <-closeDone:
		require.NoError(t, closeErr)
	case <-time.After(time.Second):
		close(release)
		<-closeDone
		t.Fatal("Close blocked on page hydration")
	}
	close(release)
	require.NoError(t, <-pullDone)
}

func TestSearchTextContinuationIDModeEnumeratesExactlyTwentyThousandMembers(t *testing.T) {
	engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	service := NewService(engine)
	t.Cleanup(func() { require.NoError(t, service.Close()) })

	nodes := make([]*storage.Node, 20_000)
	for index := range nodes {
		nodes[index] = &storage.Node{ID: storage.NodeID(fmt.Sprintf("node-%05d", index)), Labels: []string{"Document"}}
	}
	for start := 0; start < len(nodes); start += 500 {
		require.NoError(t, engine.BulkCreateNodes(nodes[start:min(start+500, len(nodes))]))
	}

	request := SearchContinuationRequest{Owner: "alice", Database: "nornic", Mode: SearchContinuationID, N: 500}
	seen := make(map[string]struct{}, len(nodes))
	for {
		page, err := service.SearchTextContinuation(context.Background(), "", DefaultSearchOptions(), request, nil, nil, nil, ChunkedSearchErrorPolicy{})
		require.NoError(t, err)
		for _, result := range page.Results {
			_, duplicate := seen[result.ID]
			require.False(t, duplicate, "duplicate member %s", result.ID)
			seen[result.ID] = struct{}{}
		}
		if !page.HasMore {
			require.Equal(t, 20_000, *page.EligibleCount)
			break
		}
		request.QID = page.QID
	}
	require.Len(t, seen, 20_000)
}

func passageIDs(passages []SearchPassage) []string {
	ids := make([]string, len(passages))
	for index := range passages {
		ids[index] = passages[index].ID
	}
	return ids
}

func BenchmarkSearchTextContinuationCompleteBuild(b *testing.B) {
	b.Run("ID/20000_members", func(b *testing.B) {
		engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
		service := NewService(engine)
		b.Cleanup(func() { _ = service.Close() })
		nodes := make([]*storage.Node, 20_000)
		for index := range nodes {
			nodes[index] = &storage.Node{ID: storage.NodeID(fmt.Sprintf("node-%05d", index)), Labels: []string{"Document"}}
		}
		for start := 0; start < len(nodes); start += 500 {
			if err := engine.BulkCreateNodes(nodes[start:min(start+500, len(nodes))]); err != nil {
				b.Fatal(err)
			}
		}
		request := SearchContinuationRequest{Owner: "benchmark", Database: "nornic", Mode: SearchContinuationID, N: 500}
		b.ReportAllocs()
		b.ReportMetric(float64(len(nodes)), "members/op")
		b.ResetTimer()
		for b.Loop() {
			page, err := service.SearchTextContinuation(context.Background(), "", DefaultSearchOptions(), request, nil, nil, nil, ChunkedSearchErrorPolicy{})
			if err != nil {
				b.Fatal(err)
			}
			if page.QID != "" {
				request.QID, request.Discard = page.QID, true
				if _, err := service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{}); err != nil {
					b.Fatal(err)
				}
				request.QID, request.Discard = "", false
			}
		}
	})

	b.Run("ID_grouped/1000_passages", func(b *testing.B) {
		engine := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
		service := NewService(engine)
		b.Cleanup(func() { _ = service.Close() })
		nodes := make([]*storage.Node, 1_000)
		for index := range nodes {
			nodes[index] = &storage.Node{
				ID: storage.NodeID(fmt.Sprintf("frame-%04d", index)), Labels: []string{"Frame"},
				Properties: map[string]any{"asset_id": "asset-a"},
			}
		}
		for start := 0; start < len(nodes); start += 500 {
			if err := engine.BulkCreateNodes(nodes[start:min(start+500, len(nodes))]); err != nil {
				b.Fatal(err)
			}
		}
		request := SearchContinuationRequest{Owner: "benchmark", Database: "nornic", Mode: SearchContinuationID, GroupBy: "asset_id", N: 1}
		b.ReportAllocs()
		b.ReportMetric(float64(len(nodes)), "passages/op")
		b.ResetTimer()
		for b.Loop() {
			if _, err := service.SearchTextContinuation(context.Background(), "", DefaultSearchOptions(), request, nil, nil, nil, ChunkedSearchErrorPolicy{}); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkSearchTextContinuationScanPath(b *testing.B) {
	base := storage.NewNamespacedEngine(storage.NewMemoryEngine(), "nornic")
	nodes := make([]*storage.Node, 20_000)
	for index := range nodes {
		nodes[index] = &storage.Node{ID: storage.NodeID(fmt.Sprintf("scan-%05d", index)), Labels: []string{"Document"}}
	}
	for start := 0; start < len(nodes); start += 500 {
		if err := base.BulkCreateNodes(nodes[start:min(start+500, len(nodes))]); err != nil {
			b.Fatal(err)
		}
	}
	benchmarks := []struct {
		name   string
		engine storage.Engine
	}{
		{name: "native_streaming", engine: base},
		{name: "AllNodes_fallback", engine: nonStreamingContinuationEngine{Engine: base}},
	}
	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			service := NewService(benchmark.engine)
			b.Cleanup(func() { _ = service.Close() })
			request := SearchContinuationRequest{Owner: "benchmark", Database: "nornic", Mode: SearchContinuationID, N: 500}
			b.ReportAllocs()
			b.ReportMetric(float64(len(nodes)), "nodes/op")
			for b.Loop() {
				page, err := service.SearchTextContinuation(context.Background(), "", DefaultSearchOptions(), request, nil, nil, nil, ChunkedSearchErrorPolicy{})
				if err != nil {
					b.Fatal(err)
				}
				if page.QID != "" {
					request.QID, request.Discard = page.QID, true
					if _, err := service.SearchTextContinuation(context.Background(), "", nil, request, nil, nil, nil, ChunkedSearchErrorPolicy{}); err != nil {
						b.Fatal(err)
					}
					request.QID, request.Discard = "", false
				}
			}
		})
	}
}
