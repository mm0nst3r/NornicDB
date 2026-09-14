package nornicdb

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/orneryd/nornicdb/pkg/cypher"
	"github.com/orneryd/nornicdb/pkg/embed"
	"github.com/orneryd/nornicdb/pkg/embeddingutil"
	"github.com/orneryd/nornicdb/pkg/search"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

func syntheticVoyageVector(basis int) []float32 {
	vector := make([]float32, 2048)
	vector[basis] = 1
	return vector
}
func writeSyntheticContext(w http.ResponseWriter, query bool) {
	rows := []map[string]any{{"index": 1, "text": "Second supporting passage", "embedding": syntheticVoyageVector(1)}, {"index": 0, "text": "First supporting passage", "embedding": syntheticVoyageVector(0)}}
	if query {
		rows = []map[string]any{{"index": 0, "embedding": syntheticVoyageVector(1)}}
	}
	w.Header().Set("x-request-id", "synthetic-context-request")
	_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"index": 0, "data": rows}}, "model": "voyage-context-4", "usage": map[string]any{"total_tokens": 20}, "chunker_version": "synthetic-1"})
}

func newSyntheticVoyage(t *testing.T, url, provider string) *embed.VoyageEmbedder {
	t.Helper()
	model := "voyage-context-4"
	if provider == "voyage-multimodal" {
		model = "voyage-multimodal-3.5"
	}
	instance, err := embed.NewVoyage(&embed.Config{Provider: provider, APIURL: url, APIKey: "synthetic-key", Model: model, Dimensions: 2048}, nil)
	require.NoError(t, err)
	return instance
}

func TestVoyageManagedWorkerPreservesDocumentsImagesAndSpaces(t *testing.T) {
	const source = "First supporting passage. Second supporting passage."
	var documents, images, queries atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		require.EqualValues(t, 2048, request["output_dimension"])
		query := request["input_type"] == "query"
		if query {
			queries.Add(1)
		}
		switch r.URL.Path {
		case "/v1/contextualizedembeddings":
			if !query {
				documents.Add(1)
				require.Equal(t, true, request["enable_auto_chunking"])
				inputs := request["inputs"].([]any)
				require.Len(t, inputs, 1)
				require.Contains(t, inputs[0].(string), source)
			}
			writeSyntheticContext(w, query)
		case "/v1/multimodalembeddings":
			inputs := request["inputs"].([]any)
			require.Len(t, inputs, 1)
			parts := inputs[0].(map[string]any)["content"].([]any)
			if !query {
				images.Add(1)
				require.Equal(t, "image_url", parts[0].(map[string]any)["type"])
				require.Equal(t, "https://example.invalid/synthetic.png", parts[0].(map[string]any)["image_url"])
			} else {
				require.Equal(t, "text", parts[0].(map[string]any)["type"])
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"index": 0, "embedding": syntheticVoyageVector(2)}}, "model": "voyage-multimodal-3.5"})
		default:
			t.Errorf("unexpected provider endpoint %s", r.URL.Path)
		}
	}))
	defer provider.Close()
	text := newSyntheticVoyage(t, provider.URL, "voyage-context")
	visual := newSyntheticVoyage(t, provider.URL, "voyage-multimodal")
	engine := storage.NewMemoryEngine()
	defer engine.Close()
	engine.SetEmbeddingsEnabled(true)
	_, err := engine.CreateNode(&storage.Node{ID: "text:document", Labels: []string{"Document"}, Properties: map[string]any{"content": source, "source_id": "source-A", "source_revision": int64(7)}})
	require.NoError(t, err)
	_, err = engine.CreateNode(&storage.Node{ID: "visual:image", Labels: []string{"Image"}, Properties: map[string]any{"source_id": "source-A", "_embedding_content": []any{map[string]any{"type": "image_url", "image_url": "https://example.invalid/synthetic.png"}}}})
	require.NoError(t, err)
	cfg := DefaultEmbedWorkerConfig()
	cfg.DeferWorkerStart = true
	cfg.BatchDelay = 0
	worker := NewEmbedWorker(text, engine, cfg)
	defer worker.Close()
	worker.SetEmbedderResolver(func(id storage.NodeID) (embed.Embedder, error) {
		if strings.HasPrefix(string(id), "visual:") {
			return visual, nil
		}
		return text, nil
	})
	require.True(t, worker.processNextBatch())
	require.True(t, worker.processNextBatch())
	document, err := engine.GetNode("text:document")
	require.NoError(t, err)
	image, err := engine.GetNode("visual:image")
	require.NoError(t, err)
	require.Len(t, document.ChunkEmbeddings, 2)
	require.Equal(t, float32(1), document.ChunkEmbeddings[1][1])
	require.Equal(t, []string{"First supporting passage", "Second supporting passage"}, document.EmbedMeta["chunk_texts"])
	require.Equal(t, "source-A", document.Properties["source_id"])
	require.EqualValues(t, 7, document.Properties["source_revision"])
	require.NotEqual(t, document.EmbedMeta["embedding_space"], image.EmbedMeta["embedding_space"])
	require.True(t, storage.ManagedEmbeddingCurrent(document))
	require.True(t, storage.ManagedEmbeddingCurrent(image), "image metadata: %#v", image.EmbedMeta)
	require.Equal(t, "completed", embeddingutil.EmbeddingWorkStatus(document)["status"])
	_, err = embed.QueryVector(context.Background(), embed.NewTracedEmbedder(embed.NewCachedEmbedder(text, 10)), "supporting text")
	require.NoError(t, err)
	_, err = embed.QueryVector(context.Background(), embed.NewCachedEmbedder(embed.NewTracedEmbedder(visual), 10), "a picture")
	require.NoError(t, err)
	require.Equal(t, int32(1), documents.Load())
	require.Equal(t, int32(1), images.Load())
	require.Equal(t, int32(2), queries.Load())

	// Actual retrieval must carry the winning chunk through parent collapse and
	// must not admit an image vector from another 2048-dimensional model.
	svc := search.NewServiceWithDimensions(engine, 2048)
	defer svc.Close()
	svc.SetEmbeddingSpace(text.EmbeddingSpace().Key())
	require.NoError(t, svc.BuildIndexes(context.Background()))
	options := search.DefaultSearchOptions()
	options.Limit = 10
	options.RerankEnabled = false
	zero := 0.0
	options.MinSimilarity = &zero
	for _, vector := range [][]float32{syntheticVoyageVector(1), nil} {
		response, err := svc.Search(context.Background(), "Second", vector, options)
		require.NoError(t, err)
		require.Len(t, response.Results, 1)
		require.Equal(t, "text:document", response.Results[0].ID)
		require.Len(t, response.Results[0].SupportingPassages, 1)
		passage := response.Results[0].SupportingPassages[0]
		require.Equal(t, 1, passage.ChunkIndex)
		require.Equal(t, "Second supporting passage", passage.Text)
		require.Equal(t, document.EmbedMeta["embedding_space"], passage.Space)
	}
	visualResponse, err := svc.Search(context.Background(), "", syntheticVoyageVector(2), options)
	require.NoError(t, err)
	for _, row := range visualResponse.Results {
		require.NotEqual(t, "visual:image", row.ID)
	}
}

func TestVoyageManagedControlsRejectInFlightResult(t *testing.T) {
	for _, action := range []string{"pause", "cancel"} {
		t.Run(action, func(t *testing.T) {
			entered := make(chan struct{})
			release := make(chan struct{})
			var calls atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if calls.Add(1) == 1 {
					close(entered)
					<-release
				}
				writeSyntheticContext(w, false)
			}))
			defer provider.Close()
			native := newSyntheticVoyage(t, provider.URL, "voyage-context")
			engine := storage.NewMemoryEngine()
			defer engine.Close()
			engine.SetEmbeddingsEnabled(true)
			namespace := storage.NewNamespacedEngine(engine, "text")
			_, err := namespace.CreateNode(&storage.Node{ID: "document", Labels: []string{"Document"}, Properties: map[string]any{"content": "A complete document"}})
			require.NoError(t, err)
			cfg := DefaultEmbedWorkerConfig()
			cfg.DeferWorkerStart = true
			cfg.BatchDelay = 0
			worker := NewEmbedWorker(native, engine, cfg)
			defer worker.Close()
			completed := make(chan struct{})
			go func() { worker.processNextBatch(); close(completed) }()
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("provider request did not start")
			}
			executor := cypher.NewStorageExecutor(namespace)
			_, controlErr := executor.Execute(context.Background(), "CALL db.embedding.control($request)", map[string]any{"request": map[string]any{"nodeId": "document", "action": action}})
			close(release)
			require.NoError(t, controlErr)
			select {
			case <-completed:
			case <-time.After(5 * time.Second):
				t.Fatal("provider request did not finish")
			}
			node, err := namespace.GetNode("document")
			require.NoError(t, err)
			require.Empty(t, node.ChunkEmbeddings)
			require.False(t, storage.NodeNeedsEmbedding(node))
			_, err = executor.Execute(context.Background(), "CALL db.embedding.control({nodeId:'document',action:'resume'})", nil)
			require.NoError(t, err)
			require.True(t, worker.processNextBatch())
			node, err = namespace.GetNode("document")
			require.NoError(t, err)
			require.Len(t, node.ChunkEmbeddings, 2, "node metadata: %#v, calls: %d", node.EmbedMeta, calls.Load())
			require.Equal(t, "completed", embeddingutil.EmbeddingWorkStatus(node)["status"])
			require.Equal(t, int32(2), calls.Load())
		})
	}
}
