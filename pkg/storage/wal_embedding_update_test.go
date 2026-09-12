package storage

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestWALEngine_UpdateNodeEmbedding_DelegatesToEmbeddingUpdater(t *testing.T) {
	base := NewMemoryEngine()

	async := NewAsyncEngine(base, DefaultAsyncEngineConfig())
	wal, err := NewWAL(t.TempDir(), nil)
	require.NoError(t, err)

	engine := NewWALEngine(async, wal)
	t.Cleanup(func() { require.NoError(t, engine.Close()) })

	node := &Node{
		ID:         NodeID("nornic:test-node"),
		Labels:     []string{"Test"},
		Properties: map[string]any{"k": "v"},
	}
	_, err = engine.CreateNode(node)
	require.NoError(t, err)

	// Make it an embedding update.
	node.ChunkEmbeddings = [][]float32{{0.1, 0.2, 0.3}}
	node.EmbedMeta = map[string]any{"has_embedding": true}

	count1, err := engine.NodeCount()
	require.NoError(t, err)

	err = engine.UpdateNodeEmbedding(node)
	require.NoError(t, err)

	count2, err := engine.NodeCount()
	require.NoError(t, err)

	// Embedding updates must never affect node counts.
	require.Equal(t, count1, count2)
}
