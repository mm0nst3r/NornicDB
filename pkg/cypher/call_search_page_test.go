package cypher

import (
	"context"
	"testing"
	"time"

	"github.com/orneryd/nornicdb/pkg/search"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

type pageCountingReranker struct{ calls int }

func (*pageCountingReranker) Name() string                     { return "page-test" }
func (*pageCountingReranker) Enabled() bool                    { return true }
func (*pageCountingReranker) IsAvailable(context.Context) bool { return true }
func (r *pageCountingReranker) Rerank(_ context.Context, _ string, candidates []search.RerankCandidate) ([]search.RerankResult, error) {
	r.calls++
	results := make([]search.RerankResult, len(candidates))
	for i, c := range candidates {
		results[i] = search.RerankResult{ID: c.ID, Content: c.Content, OriginalRank: i + 1, NewRank: i + 1, FinalScore: c.Score + 1}
	}
	return results, nil
}

func TestCallRetrievePageCompletePopulation(t *testing.T) {
	base := storage.NewMemoryEngine()
	t.Cleanup(func() { _ = base.Close() })
	engine := storage.NewNamespacedEngine(base, "page-test")
	svc := search.NewServiceWithDimensions(engine, 3)
	reranker := &pageCountingReranker{}
	svc.SetReranker(reranker)
	t.Cleanup(func() { _ = svc.Close() })
	for _, id := range []string{"a", "b", "c", "excluded"} {
		collection := "summer"
		if id == "excluded" {
			collection = "winter"
		}
		node := &storage.Node{ID: storage.NodeID(id), Labels: []string{"Document"}, Properties: map[string]any{"collection": collection, "content": "beach"}}
		if id == "a" {
			node.ChunkEmbeddings = [][]float32{{1, 0, 0}}
		}
		_, err := engine.CreateNode(node)
		require.NoError(t, err)
		require.NoError(t, svc.IndexNode(node))
	}
	exec := NewStorageExecutor(engine)
	exec.SetSearchService(svc)
	req := map[string]interface{}{"query": "beach", "embedding": []interface{}{1.0, 0.0, 0.0}, "rerank": true, "limit": 1, "pageSize": 1, "mode": "ranked_then_id", "filters": map[string]interface{}{"collection": "summer"}}
	seen := map[string]bool{}
	for i := 0; i < 3; i++ {
		result, err := exec.Execute(context.Background(), "CALL db.retrieve.page($req) YIELD page RETURN page", map[string]interface{}{"req": req})
		require.NoError(t, err)
		require.Len(t, result.Rows, 1)
		page := result.Rows[0][0].(map[string]interface{})
		require.Equal(t, int64(3), page["eligible_count"])
		hits := page["results"].([]interface{})
		require.Len(t, hits, 1)
		id := hits[0].(map[string]interface{})["id"].(string)
		require.False(t, seen[id])
		seen[id] = true
		req["cursor"] = page["next_cursor"]
		require.Equal(t, i == 2, page["collection_exhausted"])
	}
	require.Equal(t, map[string]bool{"a": true, "b": true, "c": true}, seen)
	require.Equal(t, 1, reranker.calls, "later pages must not rerank the population")
}

func TestCallRetrievePageLifecycleAndScope(t *testing.T) {
	base := storage.NewMemoryEngine()
	t.Cleanup(func() { _ = base.Close() })
	engine := storage.NewNamespacedEngine(base, "pages")
	svc := search.NewServiceWithDimensions(engine, 3)
	t.Cleanup(func() { _ = svc.Close() })
	for _, id := range []storage.NodeID{"a", "b", "c"} {
		_, err := engine.CreateNode(&storage.Node{ID: id})
		require.NoError(t, err)
	}
	exec := NewStorageExecutor(engine)
	exec.SetSearchService(svc)
	ctx := WithSearchContinuationScope(context.Background(), "alice", []string{"reader", "member"})
	ctx = WithPermissionChecker(ctx, func(string) bool { return true })
	req := map[string]interface{}{"mode": "id", "pageSize": 1}
	call := func(ctx context.Context, statement string) (*ExecuteResult, error) {
		return exec.Execute(ctx, statement, map[string]interface{}{"req": req})
	}
	first, err := call(ctx, "CALL db.retrieve.page($req)")
	require.NoError(t, err)
	page := first.Rows[0][0].(map[string]interface{})
	req["cursor"] = page["next_cursor"]
	second, err := call(ctx, "CALL db.retrieve.page($req)")
	require.NoError(t, err)
	replay, err := call(WithSearchContinuationScope(ctx, "alice", []string{"member", "reader"}), "CALL db.retrieve.page($req)")
	require.NoError(t, err)
	require.Equal(t, second.Rows, replay.Rows)
	_, err = call(WithSearchContinuationScope(ctx, "bob", []string{"reader"}), "CALL db.retrieve.page($req)")
	require.ErrorIs(t, err, search.ErrSearchCursorMismatch)
	_, err = call(WithPermissionChecker(ctx, func(string) bool { return false }), "CALL db.retrieve.page($req)")
	require.Error(t, err)
	_, err = call(ctx, "CALL db.retrieve.release($req)")
	require.NoError(t, err)
	_, err = call(ctx, "CALL db.retrieve.page($req)")
	require.ErrorIs(t, err, search.ErrSearchCursorExpired)
	delete(req, "cursor")
	require.NoError(t, svc.ConfigureSearchContinuation(search.SearchContinuationConfig{TTL: time.Nanosecond}))
	first, err = call(ctx, "CALL db.retrieve.page($req)")
	require.NoError(t, err)
	expires, err := time.Parse(time.RFC3339Nano, first.Rows[0][0].(map[string]interface{})["expires_at"].(string))
	require.NoError(t, err)
	// Windows wall-clock samples can stay equal across several quick calls.
	// Assert expiry only after the response's advertised deadline has passed.
	require.Eventually(t, func() bool { return time.Now().After(expires) }, time.Second, time.Millisecond)
	req["cursor"] = first.Rows[0][0].(map[string]interface{})["next_cursor"]
	require.NotEmpty(t, req["cursor"])
	_, err = call(ctx, "CALL db.retrieve.page($req)")
	require.ErrorIs(t, err, search.ErrSearchCursorExpired)
}

func TestCallRetrievePageEmptyAndValidation(t *testing.T) {
	base := storage.NewMemoryEngine()
	t.Cleanup(func() { _ = base.Close() })
	exec := NewStorageExecutor(storage.NewNamespacedEngine(base, "empty"))
	result, err := exec.Execute(context.Background(), "CALL db.retrieve.page({mode:'id'})", nil)
	require.NoError(t, err)
	require.Len(t, result.Rows, 1)
	page := result.Rows[0][0].(map[string]interface{})
	require.Equal(t, int64(0), page["eligible_count"])
	require.Equal(t, true, page["collection_exhausted"])
	require.Equal(t, "", page["next_cursor"])
	for _, size := range []interface{}{1.5, -1, 501, "bad"} {
		_, err := exec.Execute(context.Background(), "CALL db.retrieve.page($req)", map[string]interface{}{"req": map[string]interface{}{"mode": "id", "pageSize": size}})
		require.ErrorIs(t, err, search.ErrSearchPageRequest)
	}
	_, err = exec.Execute(WithPermissionChecker(context.Background(), func(string) bool { return true }), "CALL db.retrieve.page({mode:'id'})", nil)
	require.ErrorIs(t, err, search.ErrSearchPageRequest)
}
