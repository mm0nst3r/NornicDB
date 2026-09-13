package search

import (
	"context"
	"encoding/json"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestSearchPageBypassesCacheThroughCanonicalSearch(t *testing.T) {
	svc, engine := continuationTestService(t)
	continuationTestCreate(t, svc, engine, "a", "", "summer", nil)
	continuationTestCreate(t, svc, engine, "b", "", "summer", nil)
	options := DefaultSearchOptions()
	paging := &SearchPageOptions{Mode: SearchPageRanked, Scope: "reader", PageSize: 1}
	resolved, err := svc.resolveSearchPageRequest("beach", nil, options, paging)
	require.NoError(t, err)
	svc.resultCache = newSearchResultCache(8, time.Minute)
	key := svc.cacheNamespace + "\x00" + searchCacheKey(resolved.query, resolved.embedding, &resolved.options)
	svc.resultCache.Put(key, &SearchResponse{Status: "cached-sentinel"})
	ordinary, err := svc.Search(context.Background(), "beach", nil, &resolved.options)
	require.NoError(t, err)
	require.Equal(t, "cached-sentinel", ordinary.Status)
	page, err := svc.SearchPage(context.Background(), "beach", nil, options, paging)
	require.NoError(t, err)
	require.Equal(t, 2, page.Total)
	require.Equal(t, "cached-sentinel", svc.resultCache.Get(key).Status)
	require.False(t, options.bypassResultCache)
	paging.Cursor = page.NextCursor
	next, err := svc.SearchPage(context.Background(), "beach", nil, options, paging)
	require.NoError(t, err)
	require.Len(t, next.Results, 1)
}

func TestSearchPageMetadataProjectsOnlyAdditionalFields(t *testing.T) {
	// These stand-in extension fields test generic capture without importing any
	// optional retrieval implementation or adding them to ordinary result types.
	v := struct {
		Properties any            `json:"properties"`
		Node       any            `json:"node"`
		Vector     any            `json:"vector"`
		Score      float64        `json:"score"`
		Passages   []string       `json:"passages,omitempty"`
		Report     map[string]any `json:"rerank,omitempty"`
		Empty      []string       `json:"empty,omitempty"`
		Hidden     any            `json:"-"`
	}{Properties: make(chan int), Node: make(chan int), Vector: make(chan int), Score: math.Inf(1),
		Passages: []string{"complete original passage\nline two"}, Report: map[string]any{"status": "applied"}, Empty: []string{}, Hidden: make(chan int)}
	raw, err := searchPageMetadata(v, 4096)
	require.NoError(t, err)
	require.JSONEq(t, `{"passages":["complete original passage\nline two"],"rerank":{"status":"applied"}}`, string(raw))
	_, err = searchPageMetadata(v, int64(len(raw)-1))
	require.ErrorIs(t, err, ErrSearchContinuationLimit)
	v.Passages[0] = "mutated"
	v.Report["status"] = "mutated"
	require.NotContains(t, string(raw), "mutated")
	raw, err = searchPageMetadata(SearchResult{ID: "a", Properties: map[string]any{"unsupported": make(chan int)}}, 1)
	require.NoError(t, err)
	require.Empty(t, raw)
	raw, err = searchPageMetadata(&SearchResponse{Results: []SearchResult{{Score: math.Inf(1)}}}, 1)
	require.NoError(t, err)
	require.Empty(t, raw)
	_, err = searchPageMetadata(struct {
		Detail any `json:"detail"`
	}{make(chan int)}, 4096)
	require.ErrorIs(t, err, ErrSearchPageRequest)
}

func TestSearchPageMetadataExactBudget(t *testing.T) {
	v := struct {
		Detail string `json:"detail"`
	}{"whole payload"}
	raw, err := searchPageMetadata(v, 4096)
	require.NoError(t, err)
	exact, err := searchPageMetadata(v, int64(len(raw)))
	require.NoError(t, err)
	require.True(t, json.Valid(exact))
	require.Equal(t, raw, exact)
	_, err = searchPageMetadata(v, int64(len(raw)-1))
	require.ErrorIs(t, err, ErrSearchContinuationLimit)
}
