package cypher

import (
	"context"
	"errors"
	"testing"

	"github.com/orneryd/nornicdb/pkg/search"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

// failingNativeQueryEmbedder is a native provider whose query embedding fails,
// as a Voyage outage or rejected key would.
type failingNativeQueryEmbedder struct {
	err error
}

func (s *failingNativeQueryEmbedder) Embed(context.Context, string) ([]float32, error) {
	return []float32{1, 0}, nil
}

func (s *failingNativeQueryEmbedder) EmbedQuery(context.Context, string) ([]float32, error) {
	return nil, s.err
}

func (s *failingNativeQueryEmbedder) ChunkText(text string, maxTokens, overlap int) ([]string, error) {
	return chunkTestText(text, maxTokens, overlap)
}

func newNativeFallbackExecutor(t *testing.T) *StorageExecutor {
	t.Helper()
	ctx := context.Background()
	store := storage.NewNamespacedEngine(newTestMemoryEngine(t), "test")
	exec := NewStorageExecutor(store)
	exec.SetEmbedder(&failingNativeQueryEmbedder{err: errors.New("voyage: 401 unauthorized")})

	_, err := store.CreateNode(&storage.Node{ID: "doc", Labels: []string{"Document"}, Properties: map[string]interface{}{
		"content": "zero-trust architecture source", "embedding": []float32{1, 0},
	}})
	require.NoError(t, err)

	service := search.NewServiceWithDimensions(store, 2)
	require.NoError(t, service.BuildIndexes(ctx))
	exec.SetSearchService(service)
	return exec
}

func TestE2E_DbRetrieve_NativeQueryEmbeddingFailureFallsOpenAndReportsIt(t *testing.T) {
	exec := newNativeFallbackExecutor(t)

	result, err := exec.Execute(context.Background(), `CALL db.retrieve({query: 'zero-trust architecture', limit: 10})
YIELD node, search_method, fallback_triggered
RETURN node, search_method, fallback_triggered`, nil)
	require.NoError(t, err)
	require.Len(t, result.Rows, 1)
	node, ok := result.Rows[0][0].(*storage.Node)
	require.True(t, ok)
	require.Equal(t, "doc", string(node.ID))
	require.Equal(t, "fulltext", result.Rows[0][1])
	require.Equal(t, true, result.Rows[0][2])
}

func TestE2E_DbRetrieve_NativeQueryEmbeddingFailureHonoursFailClosed(t *testing.T) {
	exec := newNativeFallbackExecutor(t)

	_, err := exec.Execute(context.Background(), `CALL db.retrieve({query: 'zero-trust architecture', limit: 10, failClosed: true})
YIELD node RETURN node`, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "401 unauthorized")
}
