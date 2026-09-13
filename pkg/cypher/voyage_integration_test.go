package cypher

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/orneryd/nornicdb/pkg/search"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

func TestVoyageConfiguredRetrievalDoesNotHideProviderFailure(t *testing.T) {
	var calls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("x-request-id", "synthetic-auth-failure")
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer provider.Close()
	store := storage.NewNamespacedEngine(newTestMemoryEngine(t), "test")
	_, err := store.CreateNode(&storage.Node{ID: "doc", Labels: []string{"Document"}, Properties: map[string]any{"content": "alpha original passage"}})
	require.NoError(t, err)
	svc := search.NewService(store)
	require.NoError(t, svc.BuildIndexes(context.Background()))
	reranker, err := search.NewVoyageReranker(&search.CrossEncoderConfig{Enabled: true, APIURL: provider.URL + "/v1/rerank", APIKey: "synthetic-key"}, nil)
	require.NoError(t, err)
	svc.SetReranker(reranker)
	exec := NewStorageExecutor(store)
	exec.SetSearchService(svc)
	_, err = exec.Execute(context.Background(), "CALL db.rretrieve({query: 'alpha', limit: 1})", nil)
	require.Error(t, err, "configured Voyage failure must reach the caller")
	require.Equal(t, int32(1), calls.Load(), "only the actual rerank, no paid health probe")
}

func TestVoyageRerankProcedurePreservesIdentityAndReportsFallback(t *testing.T) {
	var status atomic.Int32
	status.Store(http.StatusOK)
	var calls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var input struct {
			Query      string   `json:"query"`
			Documents  []string `json:"documents"`
			Truncation bool     `json:"truncation"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&input))
		require.Equal(t, "Which document?", input.Query)
		require.Equal(t, []string{"First original text", "Second original text"}, input.Documents)
		require.True(t, input.Truncation)
		w.Header().Set("x-request-id", "synthetic-rerank")
		w.WriteHeader(int(status.Load()))
		if status.Load() == http.StatusOK {
			_, _ = w.Write([]byte(`{"data":[{"index":1,"relevance_score":0.50001},{"index":0,"relevance_score":0.5}],"model":"rerank-2.5","usage":{"total_tokens":12}}`))
		}
	}))
	defer provider.Close()
	store := storage.NewNamespacedEngine(newTestMemoryEngine(t), "test")
	svc := search.NewService(store)
	reranker, err := search.NewVoyageReranker(&search.CrossEncoderConfig{Enabled: true, APIURL: provider.URL + "/v1/rerank", APIKey: "synthetic-key"}, nil)
	require.NoError(t, err)
	svc.SetReranker(reranker)
	exec := NewStorageExecutor(store)
	exec.SetSearchService(svc)
	request := map[string]any{"query": "Which document?", "rerankTruncation": true,
		"candidates": []any{map[string]any{"id": "first", "content": "First original text", "score": 0.8}, map[string]any{"id": "second", "content": "Second original text", "score": 0.7}}}
	result, err := exec.Execute(context.Background(), "CALL db.rerank($request)", map[string]any{"request": request})
	require.NoError(t, err)
	require.Len(t, result.Rows, 2)
	require.Equal(t, "second", result.Rows[0][0])
	require.Equal(t, "Second original text", result.Rows[0][1])
	require.Equal(t, "applied", result.Rows[0][7].(map[string]any)["status"])
	status.Store(http.StatusUnauthorized)
	request["rerankFailurePolicy"] = "original"
	result, err = exec.Execute(context.Background(), "CALL db.rerank($request)", map[string]any{"request": request})
	require.NoError(t, err)
	require.Len(t, result.Rows, 2)
	require.Equal(t, "first", result.Rows[0][0])
	report := result.Rows[0][7].(map[string]any)
	require.Equal(t, "fallback", report["status"])
	require.Equal(t, "synthetic-rerank", report["metadata"].(map[string]any)["request_id"])
	require.Equal(t, int32(2), calls.Load())
}
