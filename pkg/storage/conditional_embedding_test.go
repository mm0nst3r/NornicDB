package storage

import (
	"github.com/stretchr/testify/require"
	"testing"
	"time"
)

func TestConditionalEmbeddingAcrossLocalStorageWrappers(t *testing.T) {
	for _, mode := range []string{"memory", "persistent", "wal", "async", "namespaced", "traced"} {
		t.Run(mode, func(t *testing.T) {
			var base *BadgerEngine
			var err error
			if mode == "persistent" {
				base, err = NewBadgerEngine(t.TempDir())
				require.NoError(t, err)
			} else {
				base, err = NewBadgerEngineInMemory()
				require.NoError(t, err)
			}
			var engine Engine = base
			if mode == "wal" || mode == "async" {
				wal, err := NewWAL(t.TempDir(), nil)
				require.NoError(t, err)
				engine = NewWALEngine(engine, wal)
			}
			if mode == "async" {
				engine = NewAsyncEngine(engine, DefaultAsyncEngineConfig())
			}
			id := NodeID("test:document")
			if mode == "namespaced" {
				engine = NewNamespacedEngine(engine, "test")
				id = "document"
			}
			if mode == "traced" {
				engine = NewTracedEngine(engine)
			}
			defer engine.Close()
			_, err = engine.CreateNode(&Node{ID: id, Labels: []string{"Document"}, Properties: map[string]any{"content": "original"}})
			require.NoError(t, err)
			expected, err := engine.GetNode(id)
			require.NoError(t, err)
			generated := CopyNode(expected)
			generated.ChunkEmbeddings = [][]float32{{1, 0}}
			generated.EmbedMeta = map[string]any{"embedding_operation_id": "first", "embedding_status": "completed"}
			updater, ok := engine.(ConditionalEmbeddingUpdater)
			require.True(t, ok)
			completed := make(chan error, 1)
			// Storage callbacks are permitted to read the public wrapper; a held async
			// cache lock here would deadlock normal search-index event consumers.
			base.OnNodeUpdated(func(*Node) { _, readErr := engine.GetNode(id); require.NoError(t, readErr) })
			go func() { completed <- updater.UpdateNodeEmbeddingIfCurrent(generated, expected) }()
			select {
			case err := <-completed:
				require.NoError(t, err)
			case <-time.After(5 * time.Second):
				t.Fatal("conditional update deadlocked its callback")
			}
			persisted, err := engine.GetNode(id)
			require.NoError(t, err)
			require.Equal(t, [][]float32{{1, 0}}, persisted.ChunkEmbeddings)
			stale := CopyNode(persisted)
			edited := CopyNode(persisted)
			edited.Properties["content"] = "edited"
			edited.ChunkEmbeddings = nil
			require.NoError(t, engine.UpdateNode(edited))
			require.ErrorIs(t, updater.UpdateNodeEmbeddingIfCurrent(generated, stale), ErrEmbeddingSourceChanged)
			require.NoError(t, engine.DeleteNode(id))
			require.ErrorIs(t, updater.UpdateNodeEmbeddingIfCurrent(generated, stale), ErrNotFound)
		})
	}
}
