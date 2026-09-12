package search

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

// This unit regression checks composition; real Voyage process acceptance is
// exercised separately with the provider and original Library documents.
type pageReportingReranker struct {
	testReranker
	calls int
}

func (r *pageReportingReranker) RerankWithOptions(ctx context.Context, query string, candidates []RerankCandidate, _ NativeRerankRequest) (RerankOutcome, error) {
	r.calls++
	results, err := r.Rerank(ctx, query, candidates)
	return RerankOutcome{Results: results, Report: RerankReport{Provider: "unit", Status: "applied", Candidates: len(candidates), Submitted: len(candidates), Returned: len(results)}}, err
}

func TestSearchPageIntegrationRetainsPassagesAndRerank(t *testing.T) {
	svc, engine := continuationTestService(t)
	for _, id := range []string{"a", "b"} {
		node := &storage.Node{ID: storage.NodeID(id), Labels: []string{"Document"},
			Properties:      map[string]any{"content": "whole original document " + id},
			ChunkEmbeddings: [][]float32{{1, 0, 0}, {0, 1, 0}},
			EmbedMeta:       map[string]any{"embedding_api": "contextualizedembeddings", "chunk_texts": []string{"first unrelated passage", "second matching passage " + id}}}
		_, err := engine.CreateNode(node)
		require.NoError(t, err)
		require.NoError(t, svc.IndexNode(node))
	}
	r := &pageReportingReranker{testReranker: testReranker{enabled: true}}
	svc.SetReranker(r)
	opts := DefaultSearchOptions()
	opts.Limit, opts.RerankTopK, opts.RerankEnabled = 2, 2, true
	zero := float64(0)
	opts.MinSimilarity = &zero
	paging := &SearchPageOptions{Mode: SearchPageRanked, PageSize: 1, Scope: "integration-reader"}
	for index := 0; index < 2; index++ {
		page, err := svc.SearchPage(context.Background(), "matching passage", []float32{0, 1, 0}, opts, paging)
		require.NoError(t, err)
		data, err := json.Marshal(page)
		require.NoError(t, err)
		var wire struct {
			Rerank  *RerankReport `json:"rerank"`
			Results []struct {
				Passages []SupportingPassage `json:"passages"`
			} `json:"results"`
		}
		require.NoError(t, json.Unmarshal(data, &wire))
		require.NotNil(t, wire.Rerank, "page must retain the original rerank outcome")
		require.Equal(t, "applied", wire.Rerank.Status)
		require.Equal(t, 2, wire.Rerank.Submitted)
		require.Len(t, wire.Results, 1)
		require.NotEmpty(t, wire.Results[0].Passages, "page must retain full supporting passages")
		require.Contains(t, wire.Results[0].Passages[0].Text, "matching passage")
		// Caller-owned output must not alter the frozen report or later pages.
		page.Rerank.Status = "caller changed output"
		page.Results[0].Passages[0].Text = "caller changed output"
		paging.Cursor = page.NextCursor
	}
	require.Equal(t, 1, r.calls, "later pages must not call the provider again")
}
