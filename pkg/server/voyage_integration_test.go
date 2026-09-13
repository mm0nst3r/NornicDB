package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	nornicConfig "github.com/orneryd/nornicdb/pkg/config"
	"github.com/orneryd/nornicdb/pkg/nornicdb"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

func TestVoyageHTTPAndDatabaseFailurePolicy(t *testing.T) {
	var calls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		require.Equal(t, "rerank-2.5", body["model"])
		require.Equal(t, "/v1/rerank", r.URL.Path)
		w.Header().Set("x-request-id", "synthetic-http-auth")
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer provider.Close()
	db, err := nornicdb.Open(t.TempDir(), nornicdb.DefaultConfig())
	require.NoError(t, err)
	t.Cleanup(func() { _ = db.Close() })
	cfg := DefaultConfig()
	cfg.MCPEnabled = false
	cfg.EmbeddingEnabled = false
	cfg.Features = &nornicConfig.FeatureFlagsConfig{SearchRerankEnabled: true, SearchRerankProvider: "voyage", SearchRerankModel: "rerank-2.5", SearchRerankAPIURL: provider.URL + "/v1/rerank", SearchRerankAPIKey: "synthetic-key", SearchRerankFailurePolicy: "error"}
	cfg.ProcessConfig = nornicConfig.LoadDefaults()
	cfg.ProcessConfig.Features = *cfg.Features
	server, err := New(db, nil, cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = server.Stop(context.Background()) })
	dbName := server.dbManager.DefaultDatabaseName()
	engine, err := server.dbManager.GetStorage(dbName)
	require.NoError(t, err)
	node := &storage.Node{ID: "doc", Labels: []string{"Document"}, Properties: map[string]any{"content": "alpha original text"}}
	_, err = engine.CreateNode(node)
	require.NoError(t, err)
	svc, err := server.db.EnsureSearchIndexesBuilt(context.Background(), dbName, engine)
	require.NoError(t, err)
	require.NoError(t, svc.IndexNode(node))
	response := makeRequest(t, server, http.MethodPost, "/nornicdb/search", map[string]any{"query": "alpha", "limit": 1}, "")
	require.Equal(t, http.StatusInternalServerError, response.Code, response.Body.String())
	require.Equal(t, int32(1), calls.Load())
	response = makeRequest(t, server, http.MethodPut, "/admin/databases/"+dbName+"/config", map[string]any{"overrides": map[string]string{"db.nornic.search.rerank.failure.policy": "original", "db.nornic.search.rerank.truncation": "true"}}, "")
	require.Equal(t, http.StatusOK, response.Code, response.Body.String())
	svc, err = server.db.EnsureSearchIndexesBuilt(context.Background(), dbName, engine)
	require.NoError(t, err)
	require.NoError(t, svc.IndexNode(node))
	response = makeRequest(t, server, http.MethodPost, "/nornicdb/search", map[string]any{"query": "alpha", "limit": 1}, "")
	require.Equal(t, http.StatusOK, response.Code, response.Body.String())
	var report map[string]any
	require.NoError(t, json.Unmarshal([]byte(response.Header().Get("X-NornicDB-Rerank")), &report))
	require.Equal(t, "fallback", report["status"])
	require.Contains(t, response.Body.String(), "alpha original text")
	require.Contains(t, response.Body.String(), `"status":"fallback"`)
	require.Equal(t, int32(2), calls.Load())
	// Cache an implicit HTTP Cypher executor, then change its configured policy.
	// Subsequent requests must bind the replacement search service.
	cypherBody := map[string]any{"statements": []any{map[string]any{
		"statement": "CALL db.rerank({query:'alpha',candidates:[{id:'doc',content:'alpha original text'}],limit:1})",
	}}}
	response = makeRequest(t, server, http.MethodPost, "/db/"+dbName+"/tx/commit", cypherBody, "")
	require.Equal(t, http.StatusOK, response.Code, response.Body.String())
	var before map[string]any
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &before))
	require.Empty(t, before["errors"])
	require.Contains(t, response.Body.String(), `"status":"fallback"`)
	response = makeRequest(t, server, http.MethodPut, "/admin/databases/"+dbName+"/config", map[string]any{"overrides": map[string]string{"db.nornic.search.rerank.failure.policy": "error", "db.nornic.search.rerank.truncation": "false"}}, "")
	require.Equal(t, http.StatusOK, response.Code, response.Body.String())
	response = makeRequest(t, server, http.MethodPost, "/db/"+dbName+"/tx/commit", cypherBody, "")
	var after map[string]any
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &after))
	require.NotEmpty(t, after["errors"], "cached executor retained the superseded fallback policy")
	require.Equal(t, int32(4), calls.Load())
}
