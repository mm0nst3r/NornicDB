package mcp

import (
	"context"
	"errors"
	"testing"

	"github.com/orneryd/nornicdb/pkg/nornicdb"
	"github.com/stretchr/testify/require"
)

// failingNativeQueryEmbedder is a native provider whose query embedding fails,
// as a Voyage outage or rejected key would. Document embedding still works.
type failingNativeQueryEmbedder struct {
	mockEmbedder
	queryErr error
}

func (e *failingNativeQueryEmbedder) EmbedQuery(context.Context, string) ([]float32, error) {
	return nil, e.queryErr
}

func TestHandleDiscover_NativeQueryEmbeddingFailureFallsOpenAndReportsIt(t *testing.T) {
	db, err := nornicdb.Open("", nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	_, err = db.EnsureSearchIndexesBuilt(ctx, "nornic", db.GetStorage())
	require.NoError(t, err)

	exec := db.GetCypherExecutor()
	require.NotNil(t, exec)
	_, err = exec.Execute(ctx, "CREATE (n:Memory {title: 'Alpha', content: 'alpha unique keyword content'})", nil)
	require.NoError(t, err)

	cfg := DefaultServerConfig()
	cfg.Embedder = &failingNativeQueryEmbedder{queryErr: errors.New("voyage: 401 unauthorized")}
	cfg.EmbeddingEnabled = true
	server := NewServer(db, cfg)

	raw, err := server.handleDiscover(ctx, map[string]interface{}{"query": "alpha unique keyword", "limit": 10})
	require.NoError(t, err)
	result := raw.(DiscoverResult)
	require.Equal(t, "keyword", result.Method)
	require.True(t, result.FallbackTriggered, "MCP must report that the search degraded to keyword matching")
	require.Len(t, result.Results, 1)
	require.Equal(t, "Alpha", result.Results[0].Title)
}

func TestHandleDiscover_WorkingEmbedderDoesNotReportFallback(t *testing.T) {
	db, err := nornicdb.Open("", nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	_, err = db.EnsureSearchIndexesBuilt(ctx, "nornic", db.GetStorage())
	require.NoError(t, err)

	exec := db.GetCypherExecutor()
	require.NotNil(t, exec)
	_, err = exec.Execute(ctx, "CREATE (n:Memory {title: 'Alpha', content: 'alpha unique keyword content'})", nil)
	require.NoError(t, err)

	embedding := make([]float32, 1024)
	embedding[0] = 1
	cfg := DefaultServerConfig()
	cfg.Embedder = &mockEmbedder{embedding: embedding}
	cfg.EmbeddingEnabled = true
	server := NewServer(db, cfg)
	raw, err := server.handleDiscover(ctx, map[string]interface{}{"query": "alpha unique keyword", "limit": 10})
	require.NoError(t, err)
	result := raw.(DiscoverResult)
	require.Len(t, result.Results, 1)
	require.False(t, result.FallbackTriggered)
}
