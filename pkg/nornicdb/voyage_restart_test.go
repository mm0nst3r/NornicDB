package nornicdb

import (
	"context"
	"github.com/orneryd/nornicdb/pkg/cypher"
	"github.com/orneryd/nornicdb/pkg/embeddingutil"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestVoyageDurableRetryAndSourceEditAfterFailure(t *testing.T) {
	var calls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		w.Header().Set("x-request-id", "retry-test")
		if n == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(429)
			return
		}
		if n == 3 {
			w.WriteHeader(401)
			return
		}
		writeSyntheticContext(w, false)
	}))
	defer provider.Close()
	native := newSyntheticVoyage(t, provider.URL, "voyage-context")
	dir := t.TempDir()
	engine, err := storage.NewBadgerEngine(dir)
	require.NoError(t, err)
	engine.SetEmbeddingsEnabled(true)
	_, err = engine.CreateNode(&storage.Node{ID: "text:doc", Properties: map[string]any{"content": "complete source"}})
	require.NoError(t, err)
	cfg := DefaultEmbedWorkerConfig()
	cfg.DeferWorkerStart = true
	cfg.BatchDelay = 0
	cfg.MaxRetries = 2
	worker := NewEmbedWorker(native, engine, cfg)
	require.True(t, worker.processNextBatch())
	failed, err := engine.GetNode("text:doc")
	require.NoError(t, err)
	require.Equal(t, "pending", embeddingutil.EmbeddingWorkStatus(failed)["status"])
	require.Equal(t, "retry-test", failed.EmbedMeta["embedding_request_id"])
	require.EqualValues(t, 429, failed.EmbedMeta["embedding_http_status"])
	require.False(t, storage.NodeNeedsEmbedding(failed))
	worker.Close()
	require.NoError(t, engine.Close())
	engine, err = storage.NewBadgerEngine(dir)
	require.NoError(t, err)
	defer engine.Close()
	engine.SetEmbeddingsEnabled(true)
	worker = NewEmbedWorker(native, engine, cfg)
	defer worker.Close()
	require.Eventually(t, func() bool {
		worker.processNextBatch()
		node, _ := engine.GetNode("text:doc")
		return node != nil && storage.ManagedEmbeddingCurrent(node)
	}, 8*time.Second, 50*time.Millisecond)
	got, err := engine.GetNode("text:doc")
	require.NoError(t, err)
	require.Equal(t, 2, embeddingutil.EmbeddingAttemptCount(got))
	require.Equal(t, int32(2), calls.Load())
	// A different source starts a new budget, and a terminal authentication error
	// remains visible until an explicit retry or another source change.
	got.Properties["content"] = "second source"
	require.NoError(t, engine.UpdateNode(got))
	require.True(t, worker.processNextBatch())
	got, err = engine.GetNode("text:doc")
	require.NoError(t, err)
	require.Equal(t, "failed", embeddingutil.EmbeddingWorkStatus(got)["status"])
	require.False(t, storage.NodeNeedsEmbedding(got))
	require.Equal(t, 1, embeddingutil.EmbeddingAttemptCount(got))
	got.Properties["content"] = "third source"
	require.NoError(t, engine.UpdateNode(got))
	require.True(t, worker.processNextBatch())
	got, err = engine.GetNode("text:doc")
	require.NoError(t, err)
	require.True(t, storage.ManagedEmbeddingCurrent(got))
	require.Equal(t, int32(4), calls.Load())
}

func TestVoyageInlineEmbeddingUsesCompleteManagedDocument(t *testing.T) {
	var calls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1); writeSyntheticContext(w, false) }))
	defer provider.Close()
	native := newSyntheticVoyage(t, provider.URL, "voyage-context")
	base := storage.NewMemoryEngine()
	defer base.Close()
	store := storage.NewNamespacedEngine(base, "text")
	executor := cypher.NewStorageExecutor(store)
	executor.SetEmbedder(native)
	_, err := executor.Execute(context.Background(), "CREATE (n:Document {content:'A whole document'}) WITH EMBEDDING RETURN n", nil)
	require.NoError(t, err)
	nodes, err := store.AllNodes()
	require.NoError(t, err)
	require.Len(t, nodes, 1)
	require.Len(t, nodes[0].ChunkEmbeddings, 2)
	require.True(t, storage.ManagedEmbeddingCurrent(nodes[0]))
	require.Equal(t, "completed", embeddingutil.EmbeddingWorkStatus(nodes[0])["status"])
	require.Equal(t, int32(1), calls.Load())
}
