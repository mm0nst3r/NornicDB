package search

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func voyageCandidates() []RerankCandidate {
	return []RerankCandidate{{ID: "a", Content: "Alpha full document", Score: .03}, {ID: "b", Content: "Beta full document", Score: .02}, {ID: "c", Content: "Gamma full document", Score: .01}}
}

const adapterRerankJSON = `{"data":[{"index":2,"relevance_score":0.5001},{"index":0,"relevance_score":0.5}],"model":"rerank-2.5","usage":{"total_tokens":42}}`

func adapterReranker(t *testing.T, handler http.HandlerFunc, options *VoyageRerankOptions) *VoyageReranker {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	if options == nil {
		options = &VoyageRerankOptions{ReturnCount: 2, MaxAttempts: 1}
	}
	r, err := NewVoyageReranker(&CrossEncoderConfig{Enabled: true, APIURL: server.URL + "/v1/rerank", APIKey: "synthetic-key", Model: "rerank-2.5"}, options)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestVoyageRerankerIdentityOrderingAndScores(t *testing.T) {
	r := adapterReranker(t, func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req["top_k"] != float64(2) || req["truncation"] != false || len(req["documents"].([]any)) != 3 {
			t.Fatal("wrong native payload")
		}
		_, _ = io.WriteString(w, adapterRerankJSON)
	}, nil)
	out, err := r.RerankWithOptions(context.Background(), "query", voyageCandidates(), NativeRerankRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if out.Report.Status != "applied" || out.Report.Returned != 2 || out.Report.Submitted != 3 || out.Report.Metadata.Usage.TotalTokens != 42 {
		t.Fatalf("bad report %+v", out.Report)
	}
	a, b := out.Results[0], out.Results[1]
	if a.ID != "c" || a.Content != "Gamma full document" || a.OriginalRank != 3 || a.NewRank != 1 || a.BiScore != .01 || a.FinalScore != .5001 || b.ID != "a" {
		t.Fatalf("identity/order lost %+v", out.Results)
	}
	results, err := r.Rerank(context.Background(), "query", voyageCandidates())
	if err != nil || len(results) != 2 {
		t.Fatal(err)
	}
}

func TestVoyageRerankerTopOneThresholdAndOverrides(t *testing.T) {
	r := adapterReranker(t, func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req["top_k"] != float64(1) || req["truncation"] != true {
			t.Error("overrides lost")
		}
		_, _ = io.WriteString(w, `{"data":[{"index":1,"relevance_score":0.49}]}`)
	}, nil)
	yes := true
	out, err := r.RerankWithOptions(context.Background(), "q", voyageCandidates(), NativeRerankRequest{Limit: 1, Truncation: &yes})
	if err != nil || len(out.Results) != 1 || out.Results[0].ID != "b" {
		t.Fatalf("top-one discarded: %+v %v", out, err)
	}
	out, err = r.RerankWithOptions(context.Background(), "q", voyageCandidates(), NativeRerankRequest{Limit: 1, Truncation: &yes, MinScore: .5})
	if err != nil || len(out.Results) != 0 || out.Report.Status != "applied" {
		t.Fatal("threshold incorrectly restored stage-1 results")
	}
}

func TestVoyageRerankerCandidateBudget(t *testing.T) {
	var budget atomic.Int64
	budget.Store(1)
	r := adapterReranker(t, func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		expectedBudget := int(budget.Load())
		expectedOutput := expectedBudget
		if expectedOutput > 2 {
			expectedOutput = 2
		}
		if len(req["documents"].([]any)) != expectedBudget || req["top_k"] != float64(expectedOutput) {
			t.Fatal("candidate budget/output count conflated")
		}
		if expectedOutput == 1 {
			_, _ = io.WriteString(w, `{"data":[{"index":0,"relevance_score":0.3}]}`)
		} else {
			_, _ = io.WriteString(w, `{"data":[{"index":2,"relevance_score":0.9},{"index":0,"relevance_score":0.3}]}`)
		}
	}, nil)
	r.config.TopK = 1
	out, err := r.RerankWithOptions(context.Background(), "q", voyageCandidates(), NativeRerankRequest{Limit: 10})
	if err != nil || out.Report.Candidates != 3 || out.Report.Submitted != 1 || len(out.Results) != 1 {
		t.Fatal("budget report")
	}
	budget.Store(3)
	out, err = r.RerankWithOptions(context.Background(), "q", voyageCandidates(), nativeRerankRequest(&SearchOptions{RerankTopK: 3, Limit: 10}))
	if err != nil || out.Report.Submitted != 3 || len(out.Results) != 2 || out.Results[0].ID != "c" {
		t.Fatal("explicit service budget must override the provider default without changing output count")
	}
}

func TestVoyageRerankerExplicitFailurePolicy(t *testing.T) {
	var calls atomic.Int64
	r := adapterReranker(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("X-Request-Id", "request-42")
		w.WriteHeader(429)
		_, _ = io.WriteString(w, "synthetic-key private document")
	}, nil)
	_, err := r.Rerank(context.Background(), "q", voyageCandidates())
	if err == nil {
		t.Fatal("rate limit represented as success")
	}
	out, err := r.RerankWithOptions(context.Background(), "q", voyageCandidates(), NativeRerankRequest{FailurePolicy: RerankKeepOriginal, Limit: 2})
	if err != nil || out.Report.Status != "fallback" || out.Results[0].ID != "a" || out.Results[0].FinalScore != .03 || out.Report.Metadata.StatusCode != 429 {
		t.Fatalf("explicit fallback %+v %v", out, err)
	}
	if strings.Contains(out.Report.Error, "private document") || strings.Contains(out.Report.Error, "synthetic-key") {
		t.Fatal("error leaked content")
	}
	r.options.FailurePolicy = RerankKeepOriginal
	results, err := r.Rerank(context.Background(), "q", voyageCandidates())
	if err == nil || len(results) != 2 {
		t.Fatal("legacy-shaped API concealed fallback")
	}
	_, err = r.RerankWithOptions(context.Background(), " ", voyageCandidates(), NativeRerankRequest{FailurePolicy: RerankKeepOriginal})
	if err == nil {
		t.Fatal("invalid input allowed fallback")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = r.RerankWithOptions(ctx, "q", voyageCandidates(), NativeRerankRequest{FailurePolicy: RerankKeepOriginal})
	if !errors.Is(err, context.Canceled) {
		t.Fatal("cancel ignored")
	}
	if calls.Load() != 3 {
		t.Fatalf("unexpected extra calls %d", calls.Load())
	}
}

func TestVoyageRerankerInvalidConfigAndInput(t *testing.T) {
	base := CrossEncoderConfig{Enabled: true, APIKey: "synthetic-key"}
	cfgs := []CrossEncoderConfig{{Enabled: true}, {APIKey: "k", APIURL: "https://example.invalid/embeddings"}, {APIKey: "k", TopK: -1}, {APIKey: "k", TopK: 1001}, {APIKey: "k", MinScore: math.NaN()}, {APIKey: "k", MinScore: math.Inf(1)}}
	for _, cfg := range cfgs {
		if _, err := NewVoyageReranker(&cfg, nil); err == nil {
			t.Fatal("invalid config accepted")
		}
	}
	if _, err := NewVoyageReranker(nil, nil); err == nil {
		t.Fatal("missing key accepted")
	}
	for _, opts := range []VoyageRerankOptions{{ReturnCount: -1}, {ReturnCount: 1001}, {FailurePolicy: "guess"}, {MaxAttempts: 11}} {
		if _, err := NewVoyageReranker(&base, &opts); err == nil {
			t.Fatal("invalid native options accepted")
		}
	}
	r, err := NewVoyageReranker(&base, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, req := range []NativeRerankRequest{{FailurePolicy: "guess"}, {Limit: -1}, {MinScore: math.NaN()}, {MinScore: math.Inf(1)}} {
		if _, err := r.RerankWithOptions(context.Background(), "q", nil, req); err == nil {
			t.Fatal("invalid request options accepted")
		}
	}
	if _, err := r.RerankWithOptions(nil, "q", nil, NativeRerankRequest{}); err == nil {
		t.Fatal("nil context")
	}
	for _, candidates := range [][]RerankCandidate{{{Content: "missing ID"}}, {{ID: "a", Content: "one"}, {ID: "a", Content: "two"}}} {
		if _, err := r.RerankWithOptions(context.Background(), "q", candidates, NativeRerankRequest{}); err == nil {
			t.Fatal("ambiguous ID accepted")
		}
	}
	var nilReranker *VoyageReranker
	if nilReranker.Enabled() || nilReranker.IsAvailable(context.Background()) || nilReranker.Name() != "voyage" {
		t.Fatal("nil receiver")
	}
	out, err := nilReranker.RerankWithOptions(context.Background(), "q", voyageCandidates(), NativeRerankRequest{Limit: 1})
	if err != nil || len(out.Results) != 1 || out.Report.Status != "skipped" {
		t.Fatal("disabled behavior")
	}
	out, err = r.RerankWithOptions(context.Background(), "q", nil, NativeRerankRequest{})
	if err != nil || len(out.Results) != 0 {
		t.Fatal("empty candidates")
	}
	if r.IsAvailable(nil) {
		t.Fatal("nil health context")
	}
}

func TestVoyageRerankerProbeAndConfigCopy(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req["model"] != "rerank-2.5" || req["top_k"] != float64(1) {
			t.Error("health probe contract")
		}
		_, _ = io.WriteString(w, `{"data":[{"index":0,"relevance_score":0.5}]}`)
	}))
	defer s.Close()
	cfg := CrossEncoderConfig{Enabled: true, APIKey: "synthetic-key", APIURL: s.URL + "/v1/rerank"}
	opts := VoyageRerankOptions{MaxAttempts: 1}
	r, err := NewVoyageReranker(&cfg, &opts)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Model = "mutated"
	opts.ReturnCount = 500
	if !r.IsAvailable(context.Background()) {
		t.Fatal("valid probe failed")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	time.Sleep(time.Millisecond)
	if r.IsAvailable(ctx) {
		t.Fatal("expired health request succeeded")
	}
}

func TestConfiguredNativeRerankerNeverFallsBackToOtherProvider(t *testing.T) {
	r := NewConfiguredHTTPReranker(" voyage ", &CrossEncoderConfig{Enabled: true})
	if !r.Enabled() || r.Name() != "voyage" || r.IsAvailable(context.Background()) {
		t.Fatal("invalid native config disappeared")
	}
	if _, err := r.Rerank(context.Background(), "q", voyageCandidates()); err == nil {
		t.Fatal("invalid provider succeeded")
	}
	native, ok := r.(ReportingReranker)
	if !ok {
		t.Fatal("failure lost explicit reporting capability")
	}
	out, err := native.RerankWithOptions(context.Background(), "q", voyageCandidates(), NativeRerankRequest{FailurePolicy: RerankKeepOriginal})
	if err == nil || out.Report.Status != "failed" {
		t.Fatal("invalid config failed open")
	}
	valid := NewConfiguredHTTPReranker("voyage", &CrossEncoderConfig{Enabled: true, APIKey: "synthetic-key"})
	if _, ok := valid.(*VoyageReranker); !ok {
		t.Fatal("wrong native implementation")
	}
	legacy := NewConfiguredHTTPReranker("http", &CrossEncoderConfig{Enabled: true})
	if _, ok := legacy.(*CrossEncoder); !ok {
		t.Fatal("legacy changed")
	}
}

func TestDisabledNativeFactoryDoesNotEnableMisconfiguration(t *testing.T) {
	for _, cfg := range []*CrossEncoderConfig{nil, {Enabled: false}, {Enabled: false, APIKey: "synthetic-key"}} {
		if NewConfiguredHTTPReranker("voyage", cfg).Enabled() {
			t.Fatal("disabled reranker became enabled")
		}
	}
}
