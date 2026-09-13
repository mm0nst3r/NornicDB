package search

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/orneryd/nornicdb/pkg/voyage"
)

// RerankFailurePolicy explicitly selects behavior after a native reranker failure.
// An empty policy uses the configured policy, whose default is RerankFailClosed.
type RerankFailurePolicy string

const (
	RerankFailClosed   RerankFailurePolicy = "error"
	RerankKeepOriginal RerankFailurePolicy = "original"
)

// VoyageRerankOptions supplements the existing HTTP cross-encoder configuration.
// TopK in CrossEncoderConfig remains the candidate budget; ReturnCount is Voyage's
// top_k output count. Zero ReturnCount means all submitted candidates.
type VoyageRerankOptions struct {
	ReturnCount   int
	Truncation    bool
	FailurePolicy RerankFailurePolicy
	MaxAttempts   int
	MaxRetryDelay time.Duration
}

// NativeRerankRequest carries caller overrides. Limit=0 uses provider configuration.
// A nil Truncation keeps the configured value; the default is false. MinScore=0
// disables the per-request threshold, preserving the existing SearchOptions API.
type NativeRerankRequest struct {
	// CandidateLimit overrides the configured input budget; zero keeps it.
	CandidateLimit int
	Limit          int
	MinScore       float64
	Truncation     *bool
	FailurePolicy  RerankFailurePolicy
}

// RerankReport makes an actual rerank distinguishable from explicit fallback.
// Error is a sanitized category/status string, never provider body content.
type RerankReport struct {
	Provider   string          `json:"provider"`
	Status     string          `json:"status"` // applied, fallback, skipped, failed
	Candidates int             `json:"candidates"`
	Submitted  int             `json:"submitted"`
	Returned   int             `json:"returned"`
	Error      string          `json:"error,omitempty"`
	Metadata   voyage.Metadata `json:"metadata"`
}

// RerankOutcome returns verified rankings and request-local status together.
type RerankOutcome struct {
	Results []RerankResult
	Report  RerankReport
}

// ReportingReranker is an optional extension for explicit failure policy and
// authoritative provider ordering. Legacy heuristic rerankers remain unchanged.
type ReportingReranker interface {
	Reranker
	RerankWithOptions(context.Context, string, []RerankCandidate, NativeRerankRequest) (RerankOutcome, error)
}

// VoyageReranker implements the native /v1/rerank protocol. Configuration is copied
// at construction, and every outcome carries its own diagnostics (no shared state).
type VoyageReranker struct {
	client  *voyage.Client
	config  CrossEncoderConfig
	options VoyageRerankOptions
}

var _ ReportingReranker = (*VoyageReranker)(nil)

// NewVoyageReranker reuses CrossEncoderConfig for credentials, URL, model, timeout,
// candidate budget and minimum score. APIURL is the full endpoint; empty selects
// https://api.voyageai.com/v1/rerank. Model is configurable (default rerank-2.5).
// Construction does not send a request.
//
// Example:
//
//	r, err := search.NewVoyageReranker(&search.CrossEncoderConfig{
//	    Enabled: true, APIKey: os.Getenv("VOYAGE_API_KEY"), Model: "rerank-2.5",
//	}, nil)
//	if err != nil { return err }
//	svc.SetReranker(r)
func NewVoyageReranker(config *CrossEncoderConfig, options *VoyageRerankOptions) (*VoyageReranker, error) {
	cfg := CrossEncoderConfig{}
	if config != nil {
		cfg = *config
	}
	opts := VoyageRerankOptions{Truncation: cfg.Truncation, FailurePolicy: cfg.FailurePolicy}
	if options != nil {
		opts = *options
	}
	if cfg.APIURL == "" {
		cfg.APIURL = "https://api.voyageai.com/v1/rerank"
	}
	if cfg.Model == "" {
		cfg.Model = "rerank-2.5"
	}
	if cfg.TopK == 0 {
		cfg.TopK = 100
	}
	if opts.FailurePolicy == "" {
		opts.FailurePolicy = RerankFailClosed
	}
	endpoint := strings.TrimRight(cfg.APIURL, "/")
	if !strings.HasSuffix(endpoint, "/rerank") || cfg.TopK < 1 || cfg.TopK > 1000 ||
		opts.ReturnCount < 0 || opts.ReturnCount > 1000 || !validFailurePolicy(opts.FailurePolicy) ||
		math.IsNaN(cfg.MinScore) || math.IsInf(cfg.MinScore, 0) {
		return nil, fmt.Errorf("voyage reranker: invalid configuration")
	}
	client, err := voyage.NewClient(voyage.Config{APIKey: cfg.APIKey,
		BaseURL: strings.TrimSuffix(endpoint, "/rerank"), Timeout: cfg.Timeout,
		MaxAttempts: opts.MaxAttempts, MaxRetryDelay: opts.MaxRetryDelay})
	if err != nil {
		return nil, err
	}
	return &VoyageReranker{client: client, config: cfg, options: opts}, nil
}

// Name identifies the native provider rather than the legacy generic HTTP parser.
func (r *VoyageReranker) Name() string { return "voyage" }

// Enabled reports configuration state without a network request.
func (r *VoyageReranker) Enabled() bool { return r != nil && r.client != nil && r.config.Enabled }

// IsAvailable performs a minimal real rerank request with a five-second deadline.
// Voyage has no equivalent of the legacy /health endpoint. This explicit probe can
// incur token usage; it is never called on the search hot path by this adapter.
func (r *VoyageReranker) IsAvailable(ctx context.Context) bool {
	if !r.Enabled() || ctx == nil {
		return false
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	_, err := r.client.Rerank(ctx, voyage.RerankRequest{Model: r.config.Model, Query: "health check", Documents: []string{"health check"}, TopK: 1})
	return err == nil
}

// Rerank is the legacy-shaped API. It NEVER represents provider failure as success:
// even when configured to retain original results, it returns those results with a
// non-nil error. Use RerankWithOptions to consume a fallback report explicitly.
func (r *VoyageReranker) Rerank(ctx context.Context, query string, candidates []RerankCandidate) ([]RerankResult, error) {
	out, err := r.RerankWithOptions(ctx, query, candidates, NativeRerankRequest{})
	if err != nil {
		return nil, err
	}
	if out.Report.Status == "fallback" {
		return out.Results, fmt.Errorf("voyage reranker: explicit fallback (%s)", out.Report.Error)
	}
	return out.Results, nil
}

// RerankWithOptions preserves provider order and maps every index back to the
// original ID/text/score. Fallback requires explicit policy and is reported as
// fallback, never applied. Cancellation always aborts, even under fallback policy.
func (r *VoyageReranker) RerankWithOptions(ctx context.Context, query string, candidates []RerankCandidate, request NativeRerankRequest) (RerankOutcome, error) {
	out := RerankOutcome{Report: RerankReport{Provider: "voyage", Status: "skipped", Candidates: len(candidates)}}
	policy := request.FailurePolicy
	if policy == "" && r != nil {
		policy = r.options.FailurePolicy
	}
	if policy == "" {
		policy = RerankFailClosed
	}
	if !validFailurePolicy(policy) || request.Limit < 0 || math.IsNaN(request.MinScore) || math.IsInf(request.MinScore, 0) {
		return out, fmt.Errorf("voyage reranker: invalid request options")
	}
	if ctx == nil {
		return out, fmt.Errorf("voyage reranker: context is required")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if !r.Enabled() || len(candidates) == 0 {
		out.Results = originalRerankResults(candidates, request.Limit)
		out.Report.Returned = len(out.Results)
		return out, nil
	}
	seen := make(map[string]struct{}, len(candidates))
	for _, c := range candidates {
		if c.ID == "" {
			return out, fmt.Errorf("voyage reranker: candidate ID is required")
		}
		if _, ok := seen[c.ID]; ok {
			return out, fmt.Errorf("voyage reranker: duplicate candidate IDs")
		}
		seen[c.ID] = struct{}{}
	}
	submitted := candidates
	budget := request.CandidateLimit
	if budget == 0 {
		budget = r.config.TopK
	}
	if budget < 1 || budget > 1000 {
		return out, fmt.Errorf("voyage reranker: invalid candidate budget")
	}
	if len(submitted) > budget {
		submitted = submitted[:budget]
	}
	count := r.options.ReturnCount
	if request.Limit > 0 && (count == 0 || request.Limit < count) {
		count = request.Limit
	}
	if count == 0 || count > len(submitted) {
		count = len(submitted)
	}
	truncation := r.options.Truncation
	if request.Truncation != nil {
		truncation = *request.Truncation
	}
	documents := make([]string, len(submitted))
	for i, c := range submitted {
		documents[i] = c.Content
	}
	out.Report.Submitted = len(submitted)
	response, err := r.client.Rerank(ctx, voyage.RerankRequest{Query: query, Documents: documents,
		Model: r.config.Model, TopK: count, Truncation: truncation, ReturnDocuments: false})
	if err != nil {
		var providerError *voyage.Error
		if errors.As(err, &providerError) {
			out.Report.Metadata = providerError.Metadata
		}
		out.Report.Status = "failed"
		out.Report.Error = err.Error() // voyage.Error is deliberately body/URL/secret-free.
		if policy != RerankKeepOriginal || ctx.Err() != nil || errors.Is(err, context.Canceled) {
			return out, err
		}
		// Invalid inputs/configuration are programmer errors, not transient fallback.
		if providerError != nil && (providerError.Kind == voyage.InvalidInput || providerError.Kind == voyage.InvalidConfig) {
			return out, err
		}
		out.Report.Status = "fallback"
		out.Results = originalRerankResults(candidates, count)
		out.Report.Returned = len(out.Results)
		return out, nil
	}
	out.Report.Metadata = response.Metadata
	out.Report.Status = "applied"
	threshold := r.config.MinScore
	if request.MinScore > threshold {
		threshold = request.MinScore
	}
	out.Results = make([]RerankResult, 0, len(response.Rankings))
	for _, row := range response.Rankings {
		if threshold > 0 && row.Score < threshold {
			continue
		}
		c := submitted[row.Index]
		out.Results = append(out.Results, RerankResult{ID: c.ID, Content: c.Content, OriginalRank: row.Index + 1,
			NewRank: len(out.Results) + 1, BiScore: c.Score, CrossScore: row.Score, FinalScore: row.Score})
	}
	out.Report.Returned = len(out.Results)
	return out, nil
}

func validFailurePolicy(p RerankFailurePolicy) bool {
	return p == RerankFailClosed || p == RerankKeepOriginal
}

func originalRerankResults(candidates []RerankCandidate, limit int) []RerankResult {
	if limit > 0 && len(candidates) > limit {
		candidates = candidates[:limit]
	}
	out := make([]RerankResult, len(candidates))
	for i, c := range candidates {
		out[i] = RerankResult{ID: c.ID, Content: c.Content, OriginalRank: i + 1, NewRank: i + 1, BiScore: c.Score, CrossScore: c.Score, FinalScore: c.Score}
	}
	return out
}

// NewConfiguredHTTPReranker selects native Voyage for provider="voyage", preserving
// the existing generic HTTP implementation for other providers. A bad Voyage
// configuration returns an enabled, explicitly failing reporting reranker rather
// than nil, so per-database resolver fallback cannot silently select another model.
// Call NewVoyageReranker directly when construction-time errors can be returned.
func NewConfiguredHTTPReranker(provider string, config *CrossEncoderConfig) Reranker {
	if !strings.EqualFold(strings.TrimSpace(provider), "voyage") {
		return NewCrossEncoder(config)
	}
	r, err := NewVoyageReranker(config, nil)
	if err != nil {
		return &invalidVoyageReranker{cause: err, enabled: config != nil && config.Enabled}
	}
	return r
}

type invalidVoyageReranker struct {
	cause   error
	enabled bool
}

func (r *invalidVoyageReranker) Name() string                     { return "voyage" }
func (r *invalidVoyageReranker) Enabled() bool                    { return r.enabled }
func (r *invalidVoyageReranker) IsAvailable(context.Context) bool { return false }
func (r *invalidVoyageReranker) Rerank(context.Context, string, []RerankCandidate) ([]RerankResult, error) {
	return nil, r.cause
}
func (r *invalidVoyageReranker) RerankWithOptions(_ context.Context, _ string, candidates []RerankCandidate, _ NativeRerankRequest) (RerankOutcome, error) {
	return RerankOutcome{Report: RerankReport{Provider: "voyage", Status: "failed", Candidates: len(candidates), Error: r.cause.Error()}}, r.cause
}

// Map returns an owned JSON-compatible report for Cypher/Bolt and other adapters.
// It contains no request content, endpoint, or credential.
func (r *RerankReport) Map() map[string]any {
	if r == nil {
		return nil
	}
	data, _ := json.Marshal(r)
	var result map[string]any
	_ = json.Unmarshal(data, &result)
	return result
}
