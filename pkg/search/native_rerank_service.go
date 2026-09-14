package search

import (
	"context"
	"fmt"
	"strings"

	"github.com/orneryd/nornicdb/pkg/storage"
)

// errRequiredRerank marks a failure that retrieval fallback MUST NOT swallow.
// This is deliberately separate from "no retrieval results".
type errRequiredRerank struct{ cause error }

func (e *errRequiredRerank) Error() string { return "required reranking failed: " + e.cause.Error() }
func (e *errRequiredRerank) Unwrap() error { return e.cause }

func (s *Service) reportingReranker() ReportingReranker {
	s.mu.RLock()
	defer s.mu.RUnlock()
	r, _ := s.reranker.(ReportingReranker)
	if r != nil && r.Enabled() {
		return r
	}
	return nil
}

// applyReportingRerank is the native Stage-2 seam: no flat-score heuristic, no
// success-on-error, no invented IDs, and no provider scores assigned to fallback.
func (s *Service) applyReportingRerank(ctx context.Context, query string, results []rrfResult, opts *SearchOptions, seenOrphans map[string]bool, r ReportingReranker) ([]rrfResult, *RerankReport, error) {
	if len(results) == 0 {
		return results, nil, nil
	}
	candidates := make([]RerankCandidate, 0, len(results))
	byID := make(map[string]rrfResult, len(results))
	for _, result := range results {
		node, err := s.engine.GetNode(storage.NodeID(result.ID))
		if err != nil {
			if s.handleOrphanedEmbedding(ctx, result.ID, err, seenOrphans) {
				continue
			}
			// Never echo a storage error body: it may include property content.
			return nil, nil, &errRequiredRerank{fmt.Errorf("unable to load rerank candidate")}
		}
		if node == nil {
			continue
		}
		content := s.extractSearchableText(node)
		if len(result.SupportingPassages) > 0 {
			texts := make([]string, len(result.SupportingPassages))
			for i, p := range result.SupportingPassages {
				texts[i] = p.Text
			}
			content = strings.Join(texts, "\n\n")
		}
		if content == "" {
			continue
		}
		candidates = append(candidates, RerankCandidate{ID: result.ID, Content: content, Score: result.RRFScore})
		byID[result.ID] = result
	}
	out, err := r.RerankWithOptions(ctx, query, candidates, nativeRerankRequest(opts))
	if err != nil {
		return nil, &out.Report, &errRequiredRerank{err}
	}
	if out.Report.Status == "skipped" {
		return results, &out.Report, nil
	}
	converted := make([]rrfResult, 0, len(out.Results))
	for _, row := range out.Results {
		original, ok := byID[row.ID]
		if !ok {
			return nil, &out.Report, &errRequiredRerank{fmt.Errorf("reranker returned an unknown candidate")}
		}
		original.RRFScore = row.FinalScore
		converted = append(converted, original)
	}
	return converted, &out.Report, nil
}

// RerankCandidatesWithReport applies the canonical reranker to supplied candidates
// and exposes the provider outcome. Native request overrides match Search.
// Legacy rerankers retain their existing contract and return a nil report.
func (s *Service) RerankCandidatesWithReport(ctx context.Context, query string, candidates []RerankCandidate, opts *SearchOptions) ([]RerankResult, *RerankReport, error) {
	if s == nil {
		return nil, nil, fmt.Errorf("search service is unavailable")
	}
	if opts == nil {
		opts = DefaultSearchOptions()
	}
	native := s.reportingReranker()
	if native == nil {
		rows, err := s.RerankCandidates(ctx, query, candidates, opts)
		return rows, nil, err
	}
	out, err := native.RerankWithOptions(ctx, query, candidates, nativeRerankRequest(opts))
	return out.Results, &out.Report, err
}

func nativeRerankRequest(opts *SearchOptions) NativeRerankRequest {
	return NativeRerankRequest{CandidateLimit: opts.RerankTopK, Limit: opts.Limit, MinScore: opts.RerankMinScore,
		Truncation: opts.RerankTruncation, FailurePolicy: opts.RerankFailurePolicy}
}

// NativeRerankEnabled reports configured native reranking without probing a provider.
func (s *Service) NativeRerankEnabled() bool { return s != nil && s.reportingReranker() != nil }

// RerankSearchResponse applies native reranking once after an adapter combines
// query-chunk retrieval. It preserves the original result fields and RRF ranks.
// Retrieval for those chunks must have RerankEnabled=false.
func (s *Service) RerankSearchResponse(ctx context.Context, query string, response *SearchResponse, opts *SearchOptions) error {
	if response == nil || opts == nil || !opts.RerankEnabled {
		return nil
	}
	native := s.reportingReranker()
	if native == nil {
		return nil
	}
	fused := make([]rrfResult, len(response.Results))
	originals := make(map[string]SearchResult, len(response.Results))
	for i, row := range response.Results {
		fused[i] = rrfResult{ID: row.ID, RRFScore: row.Score, VectorRank: row.VectorRank, BM25Rank: row.BM25Rank, OriginalScore: row.Similarity, SupportingPassages: row.SupportingPassages}
		originals[row.ID] = row
	}
	ranked, report, err := s.applyReportingRerank(ctx, query, fused, opts, make(map[string]bool), native)
	response.Rerank = report
	if err != nil {
		return err
	}
	response.Results = make([]SearchResult, 0, len(ranked))
	for _, row := range ranked {
		result := originals[row.ID]
		result.Score = row.RRFScore
		response.Results = append(response.Results, result)
	}
	response.Returned = len(response.Results)
	if report != nil && report.Status == "applied" {
		response.SearchMethod += "+rerank"
	}
	if report != nil && report.Status == "fallback" {
		response.SearchMethod += "+rerank_fallback"
		response.FallbackTriggered = true
	}
	return nil
}
