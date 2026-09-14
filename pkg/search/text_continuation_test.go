package search

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

func TestSearchTextContinuationExpandsWithoutReembedding(t *testing.T) {
	service := NewService(storage.NewMemoryEngine())
	t.Cleanup(func() { require.NoError(t, service.Close()) })

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
			results[index] = SearchResult{ID: fmt.Sprintf("node-%03d", index), Score: float64(opts.Limit - index)}
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
	require.Equal(t, int32(1), chunkCalls.Load())
	require.Equal(t, int32(2), embedCalls.Load())
}
