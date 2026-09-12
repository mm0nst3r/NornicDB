package nornicgrpc

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	gen "github.com/orneryd/nornicdb/pkg/nornicgrpc/gen"
	"github.com/orneryd/nornicdb/pkg/search"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/test/bufconn"
)

func TestVoyageGRPCFusesThenReranksOnceAndReports(t *testing.T) {
	var calls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var input struct {
			Query     string   `json:"query"`
			Documents []string `json:"documents"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&input))
		require.Equal(t, "alpha beta", input.Query)
		require.NotEmpty(t, input.Documents)
		w.Header().Set("x-request-id", "synthetic-grpc")
		_, _ = fmt.Fprintf(w, `{"data":[{"index":%d,"relevance_score":0.9}],"model":"rerank-2.5"}`, len(input.Documents)-1)
	}))
	defer provider.Close()
	base := storage.NewMemoryEngine()
	defer base.Close()
	engine := storage.NewNamespacedEngine(base, "test")
	for _, id := range []string{"alpha", "beta"} {
		_, err := engine.CreateNode(&storage.Node{ID: storage.NodeID(id), Labels: []string{"Document"}, Properties: map[string]any{"content": id + " original passage"}, ChunkEmbeddings: [][]float32{{1, 0}}})
		require.NoError(t, err)
	}
	retrieval := search.NewServiceWithDimensions(engine, 2)
	require.NoError(t, retrieval.BuildIndexes(context.Background()))
	reranker, err := search.NewVoyageReranker(&search.CrossEncoderConfig{Enabled: true, APIURL: provider.URL + "/v1/rerank", APIKey: "synthetic-key"}, nil)
	require.NoError(t, err)
	retrieval.SetReranker(reranker)
	service, err := NewService(Config{RerankEnabled: true}, func(context.Context, string) ([]float32, error) { return []float32{1, 0}, nil }, func(context.Context, string) ([]string, error) { return []string{"alpha", "beta"}, nil }, retrieval)
	require.NoError(t, err)
	listener := bufconn.Listen(1 << 20)
	defer listener.Close()
	server := grpc.NewServer()
	defer server.Stop()
	gen.RegisterNornicSearchServer(server, service)
	go func() { _ = server.Serve(listener) }()
	client, err := grpc.NewClient("passthrough:///voyage-test", grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }))
	require.NoError(t, err)
	defer client.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	var trailer metadata.MD
	response, err := gen.NewNornicSearchClient(client).SearchText(ctx, &gen.SearchTextRequest{Query: "alpha beta", Limit: 1}, grpc.Trailer(&trailer))
	require.NoError(t, err)
	require.Len(t, response.Hits, 1)
	require.Equal(t, int32(1), calls.Load())
	require.Equal(t, "chunked_rrf_hybrid+rerank", response.SearchMethod)
	require.Len(t, trailer.Get("nornicdb-rerank"), 1)
	var report map[string]any
	require.NoError(t, json.Unmarshal([]byte(trailer.Get("nornicdb-rerank")[0]), &report))
	require.Equal(t, "applied", report["status"])
	require.Equal(t, "synthetic-grpc", report["metadata"].(map[string]any)["request_id"])
}
