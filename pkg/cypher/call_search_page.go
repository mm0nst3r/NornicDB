package cypher

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/orneryd/nornicdb/pkg/search"
)

type searchContinuationScopeKey struct{}

// SearchContinuationErrorCode maps continuation state errors consistently for
// native Cypher transports. Invalidated/expired cursors require a fresh query,
// so these are client errors rather than automatic transaction-retry signals.
func SearchContinuationErrorCode(err error) string {
	for _, item := range []struct {
		cause  error
		suffix string
	}{
		{search.ErrSearchCursorInvalidated, "SearchCursorInvalidated"},
		{search.ErrSearchCursorExpired, "SearchCursorExpired"},
		{search.ErrSearchCursorMismatch, "SearchCursorMismatch"},
		{search.ErrSearchCursorInvalid, "SearchCursorInvalid"},
		{search.ErrSearchContinuationLimit, "SearchContinuationLimit"},
		{search.ErrSearchContinuationClosed, "SearchContinuationClosed"},
		{search.ErrSearchPageRequest, "SearchPageRequestInvalid"},
	} {
		if errors.Is(err, item.cause) {
			return "Neo.ClientError.Statement." + item.suffix
		}
	}
	return ""
}

// WithSearchContinuationScope binds native continuation to a trusted principal
// and its current roles. Transports must derive these from authentication, never
// from query arguments. Normal query authorization still runs on every request.
func WithSearchContinuationScope(ctx context.Context, principal string, roles []string) context.Context {
	owned := append([]string(nil), roles...)
	sort.Strings(owned)
	encoded, _ := json.Marshal(struct {
		Principal string
		Roles     []string
	}{principal, owned})
	return context.WithValue(ctx, searchContinuationScopeKey{}, string(encoded))
}

func searchContinuationScope(ctx context.Context) (string, error) {
	if scope, ok := ctx.Value(searchContinuationScopeKey{}).(string); ok {
		return scope, nil
	}
	if ctx.Value(permissionCheckerKey{}) != nil || ctx.Value(databaseAuthorizationKey{}) != nil {
		return "", fmt.Errorf("search continuation requires a trusted principal scope: %w", search.ErrSearchPageRequest)
	}
	return "embedded", nil
}

func (e *StorageExecutor) callDbRetrievePage(ctx context.Context, statement string, release bool) (*ExecuteResult, error) {
	name := "DB.RETRIEVE.PAGE"
	if release {
		name = "DB.RETRIEVE.RELEASE"
	}
	req, err := e.parseRagProcedureRequest(ctx, statement, name)
	if err != nil {
		return nil, err
	}
	query := stringOr(req["query"], stringOr(req["text"], ""))
	opts, failClosed, err := retrievalOptions(query, req, false)
	if err != nil {
		return nil, err
	}
	if raw, present := policyPresent(req, "rerank", "rerankEnabled", "rerank_enabled"); present {
		enabled, ok := toBool(raw)
		if !ok {
			return nil, search.ErrSearchPageRequest
		}
		opts.RerankEnabled = enabled
	}
	// Embeddings are explicit and generated once by the caller. The existing
	// db.index.vector.embed procedure supports configured providers when needed.
	// Continuation therefore never calls an embedding provider on later pages.
	var embedding []float32
	if raw, present := policyPresent(req, "embedding", "queryEmbedding", "query_embedding"); present {
		embedding, err = parseRetrieveEmbedding(raw, true)
		if err != nil {
			return nil, err
		}
	}
	mode := search.SearchPageMode(stringOr(req["mode"], ""))
	if failClosed && mode != search.SearchPageID && len(embedding) == 0 {
		return nil, errRetrieveEmbeddingInvalid
	}
	scope, err := searchContinuationScope(ctx)
	if err != nil {
		return nil, err
	}
	paging := &search.SearchPageOptions{Mode: mode, Cursor: stringOr(req["cursor"], ""), GroupBy: stringOr(firstPresent(req, "groupBy", "group_by"), ""), Scope: scope}
	if raw, present := policyPresent(req, "pageSize", "page_size"); present {
		n, ok := ragToFloat64(raw)
		if !ok || !isFinite(n) || n < 0 || n != math.Trunc(n) || n > float64(math.MaxInt32) {
			return nil, search.ErrSearchPageRequest
		}
		paging.PageSize = int(n)
	}
	svc := e.searchService
	if svc == nil {
		dimensions := search.DefaultVectorDimensions
		if len(embedding) > 0 {
			dimensions = len(embedding)
		}
		svc = search.NewServiceWithDimensions(e.storage, dimensions)
		e.searchService = svc
	}
	if release {
		if err := svc.ReleaseSearchPage(query, embedding, opts, paging); err != nil {
			return nil, err
		}
		return &ExecuteResult{Columns: []string{"released"}, Rows: [][]interface{}{{true}}}, nil
	}
	page, err := svc.SearchPage(ctx, query, embedding, opts, paging)
	if err != nil {
		return nil, err
	}
	return &ExecuteResult{Columns: []string{"page"}, Rows: [][]interface{}{{nativeSearchPage(page)}}}, nil
}

func nativeSearchPage(p *search.SearchPageResponse) map[string]interface{} {
	hits := make([]interface{}, 0, len(p.Results))
	for _, h := range p.Results {
		hit := map[string]interface{}{"id": h.ID, "group_key": h.GroupKey, "phase": string(h.Phase), "score": h.Score, "similarity": h.Similarity, "rrf_score": h.RRFScore, "vector_rank": int64(h.VectorRank), "bm25_rank": int64(h.BM25Rank)}
		if len(h.Passages) > 0 {
			hit["passages"] = search.PassageMaps(h.Passages)
		}
		hits = append(hits, hit)
	}
	var eligible interface{}
	if p.EligibleCount != nil {
		eligible = int64(*p.EligibleCount)
	}
	result := map[string]interface{}{
		"total_candidates": int64(p.TotalCandidates), "fallback_triggered": p.FallbackTriggered,
		"vector_stop_reason": p.VectorStopReason, "vector_candidate_limit": int64(p.VectorCandidateLimit),
		"bm25_stop_reason": p.BM25StopReason, "bm25_candidate_limit": int64(p.BM25CandidateLimit),
		"results": hits, "next_cursor": p.NextCursor, "returned": int64(p.Returned), "position": int64(p.Position), "total": int64(p.Total), "ranked_count": int64(p.RankedCount), "eligible_count": eligible,
		"mode": string(p.Mode), "grouped": p.Grouped, "population": p.Population, "search_method": p.SearchMethod, "candidate_limit": int64(p.CandidateLimit),
		"ranked_pool_exhausted": p.RankedPoolExhausted, "exhausted": p.Exhausted, "collection_exhausted": p.CollectionExhausted, "completion": p.Completion, "expires_at": p.ExpiresAt.UTC().Format(time.RFC3339Nano),
	}
	if p.Rerank != nil {
		result["rerank"] = p.Rerank.Map()
	}
	return result
}
