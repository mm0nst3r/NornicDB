package replication

import (
	"context"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
	"testing"
	"time"
)

func TestManagedConditionalReplicationPreservesSourceAndControl(t *testing.T) {
	base := storage.NewMemoryEngine()
	defer base.Close()
	adapter, err := NewStorageAdapterWithWAL(base, t.TempDir())
	require.NoError(t, err)
	defer adapter.Close()
	replicator := NewStandaloneReplicator(&Config{}, adapter)
	require.NoError(t, replicator.Start(context.Background()))
	replicated := NewReplicatedEngine(base, replicator, time.Second)
	namespace := storage.NewNamespacedEngine(replicated, "text")
	_, err = namespace.CreateNode(&storage.Node{ID: "doc", Properties: map[string]any{"content": "source"}})
	require.NoError(t, err)
	expected, err := namespace.GetNode("doc")
	require.NoError(t, err)
	proposed := storage.CopyNode(expected)
	proposed.ChunkEmbeddings = [][]float32{{1, 0}, {0, 1}}
	proposed.EmbedMeta = map[string]any{"embedding_status": "completed", "chunk_texts": []string{"first", "second"}, "embedding_attempts": []map[string]any{{"operation_id": "test", "outcome": "completed"}}}
	require.NoError(t, namespace.UpdateNodeEmbeddingIfCurrent(proposed, expected))
	got, err := namespace.GetNode("doc")
	require.NoError(t, err)
	require.Equal(t, proposed.ChunkEmbeddings, got.ChunkEmbeddings)
	require.Equal(t, proposed.EmbedMeta, got.EmbedMeta)
	edit := storage.CopyNode(got)
	edit.Properties["content"] = "edited"
	edit.EmbedMeta["embedding_control"] = "paused"
	require.NoError(t, namespace.UpdateNode(edit))
	require.ErrorIs(t, namespace.UpdateNodeEmbeddingIfCurrent(proposed, expected), storage.ErrEmbeddingSourceChanged)
	got, err = namespace.GetNode("doc")
	require.NoError(t, err)
	require.Equal(t, "edited", got.Properties["content"])
	require.Equal(t, "paused", got.EmbedMeta["embedding_control"])
}
