package voyage

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const testKey = "synthetic-secret-never-real"

type roundTrip func(*http.Request) (*http.Response, error)

func (f roundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func testClient(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	s := httptest.NewServer(handler)
	t.Cleanup(s.Close)
	c, err := NewClient(Config{APIKey: testKey, BaseURL: s.URL + "/v1", MaxAttempts: 1})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func requireKind(t *testing.T, err error, kind string) *Error {
	t.Helper()
	var e *Error
	if !errors.As(err, &e) || e.Kind != kind {
		t.Fatalf("want kind %s, got %v", kind, err)
	}
	return e
}

func rerankRequest() RerankRequest {
	return RerankRequest{Model: "rerank-2.5", Query: "synthetic query", Documents: []string{"alpha", "beta", "gamma"}, TopK: 2}
}

const goodRerank = `{"data":[{"index":2,"relevance_score":0.5001},{"index":0,"relevance_score":0.5}],"model":"rerank-2.5","usage":{"total_tokens":100}}`

func TestClientConfiguration(t *testing.T) {
	c, err := NewClient(Config{APIKey: testKey})
	if err != nil || c.baseURL != "https://api.voyageai.com/v1" || c.attempts != 3 || c.http.Timeout != 30*time.Second {
		t.Fatalf("defaults: %v %+v", err, c)
	}
	cases := []Config{
		{}, {APIKey: "  "}, {APIKey: "secret\r\nHeader:x"},
		{APIKey: testKey, BaseURL: "://bad"},
		{APIKey: testKey, BaseURL: "https:///missing"},
		{APIKey: testKey, BaseURL: "https://user:password@api.voyageai.com/v1"},
		{APIKey: testKey, BaseURL: "https://api.voyageai.com/v1?key=secret"},
		{APIKey: testKey, BaseURL: "https://api.voyageai.com/v1?"},
		{APIKey: testKey, BaseURL: "https://api.voyageai.com/v1#fragment"},
		{APIKey: testKey, BaseURL: "mailto:someone@example.invalid"},
		{APIKey: testKey, BaseURL: "http://example.invalid/v1"},
		{APIKey: testKey, BaseURL: "ftp://localhost/v1"},
		{APIKey: testKey, Timeout: -1}, {APIKey: testKey, MaxAttempts: -1},
		{APIKey: testKey, MaxAttempts: 11}, {APIKey: testKey, MaxRetryDelay: -1},
		{APIKey: testKey, MaxResponseBytes: -1}, {APIKey: testKey, MaxResponseBytes: 1 << 31},
	}
	for i, cfg := range cases {
		t.Run(fmt.Sprint(i), func(t *testing.T) { _, err := NewClient(cfg); requireKind(t, err, InvalidConfig) })
	}
	for _, base := range []string{"http://localhost:123/v1", "http://[::1]:123/v1", "https://example.invalid/v1/"} {
		if _, err := NewClient(Config{APIKey: testKey, BaseURL: base}); err != nil {
			t.Fatal(err)
		}
	}
	b, err := json.Marshal(Config{APIKey: testKey, BaseURL: "https://private.example.invalid/v1"})
	if err != nil || strings.Contains(string(b), testKey) || strings.Contains(string(b), "private") {
		t.Fatal("config serialized secrets")
	}
}

func TestRerankContract(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.URL.Path != "/v1/rerank" || r.Header.Get("Authorization") != "Bearer "+testKey || r.Header.Get("Content-Type") != "application/json" {
			t.Error("incorrect HTTP contract")
		}
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		for _, field := range []string{"top_n", "results", "scores"} {
			if _, ok := body[field]; ok {
				t.Errorf("foreign contract field %s", field)
			}
		}
		if string(body["top_k"]) != "2" || string(body["truncation"]) != "false" || string(body["return_documents"]) != "false" {
			t.Errorf("request fields %s", body)
		}
		var docs []string
		_ = json.Unmarshal(body["documents"], &docs)
		if strings.Join(docs, "|") != "alpha|beta|gamma" {
			t.Error("document text changed")
		}
		w.Header().Set("X-Request-Id", "req-123")
		_, _ = io.WriteString(w, goodRerank)
	})
	out, err := c.Rerank(context.Background(), rerankRequest())
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Rankings) != 2 || out.Rankings[0].Index != 2 || out.Rankings[1].Index != 0 || out.Rankings[0].Score != 0.5001 {
		t.Fatalf("mapping/order: %+v", out)
	}
	if out.Metadata.RequestID != "req-123" || out.Metadata.StatusCode != 200 || out.Metadata.Usage.TotalTokens != 100 || out.Metadata.Attempts != 1 {
		t.Fatalf("metadata %+v", out.Metadata)
	}
}

func TestRerankTopOneAndAll(t *testing.T) {
	for _, k := range []int{0, 1, 3} {
		t.Run(fmt.Sprint(k), func(t *testing.T) {
			req := rerankRequest()
			req.TopK = k
			req.Truncation = true
			req.ReturnDocuments = true
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				var got RerankRequest
				_ = json.NewDecoder(r.Body).Decode(&got)
				if !got.Truncation || !got.ReturnDocuments || got.TopK != k {
					t.Error("explicit options lost")
				}
				n := k
				if n == 0 {
					n = 3
				}
				rows := make([]map[string]any, n)
				for i := range rows {
					rows[i] = map[string]any{"index": 2 - i, "relevance_score": 0.5}
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"data": rows})
			})
			out, err := c.Rerank(context.Background(), req)
			if err != nil {
				t.Fatal(err)
			}
			n := k
			if n == 0 {
				n = 3
			}
			if len(out.Rankings) != n || out.Rankings[0].Index != 2 {
				t.Fatal("equal score/top-one order changed")
			}
		})
	}
}

func TestRerankInvalidInput(t *testing.T) {
	var calls atomic.Int64
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) { calls.Add(1) })
	mutations := []func(*RerankRequest){
		func(r *RerankRequest) { r.Model = "" }, func(r *RerankRequest) { r.Query = " " },
		func(r *RerankRequest) { r.Documents = nil }, func(r *RerankRequest) { r.Documents = make([]string, 1001) },
		func(r *RerankRequest) { r.TopK = -1 }, func(r *RerankRequest) { r.TopK = 4 },
		func(r *RerankRequest) { r.Documents[0] = " " },
	}
	for _, mutate := range mutations {
		req := rerankRequest()
		mutate(&req)
		_, err := c.Rerank(context.Background(), req)
		requireKind(t, err, InvalidInput)
	}
	if calls.Load() != 0 {
		t.Fatal("invalid requests made network calls")
	}
}

func TestRerankRejectsMalformedResponse(t *testing.T) {
	bodies := []string{
		`{}`, `null`, `{"data":null}`, `{"results":[{"index":2,"relevance_score":0.8}]}`,
		`{"data":[]}`, `{"data":[{"index":0,"relevance_score":0.8}]}`,
		`{"data":[{"index":0,"relevance_score":0.8},{"index":0,"relevance_score":0.7}]}`,
		`{"data":[{"index":-1,"relevance_score":0.8},{"index":0,"relevance_score":0.7}]}`,
		`{"data":[{"index":3,"relevance_score":0.8},{"index":0,"relevance_score":0.7}]}`,
		`{"data":[{"relevance_score":0.8},{"index":0,"relevance_score":0.7}]}`,
		`{"data":[{"index":2},{"index":0,"relevance_score":0.7}]}`,
		`{"data":[{"index":null,"relevance_score":0.8},{"index":0,"relevance_score":0.7}]}`,
		`{"data":[{"index":2,"relevance_score":null},{"index":0,"relevance_score":0.7}]}`,
		`{"data":[{"index":2,"relevance_score":1e999},{"index":0,"relevance_score":0.7}]}`,
		`{"data":[{"index":"2","relevance_score":0.8},{"index":0,"relevance_score":0.7}]}`,
		goodRerank + ` {"other":1}`, goodRerank + ` garbage`,
		strings.Replace(goodRerank, "rerank-2.5", "wrong-model", 1),
		strings.Replace(goodRerank, `"total_tokens":100`, `"total_tokens":-1`, 1),
	}
	for i, body := range bodies {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, body) })
			out, err := c.Rerank(context.Background(), rerankRequest())
			requireKind(t, err, MalformedResponse)
			if out != nil {
				t.Fatal("partial success on malformed response")
			}
		})
	}
}

func TestStatusesAndPrivacy(t *testing.T) {
	for _, status := range []int{400, 401, 403, 404, 429, 500, 502, 503, 599} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Request-Id", "request-safe")
				w.Header().Set("Retry-After", "12")
				w.WriteHeader(status)
				_, _ = io.WriteString(w, `{"detail":"`+testKey+` synthetic query signed-image-url"}`)
			})
			_, err := c.Rerank(context.Background(), rerankRequest())
			kind := ProviderFailure
			if status == 401 || status == 403 {
				kind = Authentication
			}
			if status == 429 {
				kind = RateLimit
			}
			e := requireKind(t, err, kind)
			if e.Metadata.StatusCode != status || e.Metadata.RetryAfter != 12*time.Second || e.Metadata.RequestID != "request-safe" {
				t.Fatal("diagnostics lost")
			}
			if e.Retryable() != (status == 429 || status >= 500) {
				t.Fatal("wrong retry classification")
			}
			for _, secret := range []string{testKey, "synthetic query", "signed-image-url"} {
				if strings.Contains(fmt.Sprintf("%+v", err), secret) {
					t.Fatal("error leaked content")
				}
			}
			if errors.Unwrap(e) != nil {
				t.Fatal("unexpected raw provider error")
			}
		})
	}
}

func TestRetryBoundariesAndCancellation(t *testing.T) {
	t.Run("retries-transient", func(t *testing.T) {
		var calls atomic.Int64
		c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
			if calls.Add(1) < 3 {
				w.WriteHeader(503)
				return
			}
			_, _ = io.WriteString(w, goodRerank)
		})
		c.attempts = 3
		out, err := c.Rerank(context.Background(), rerankRequest())
		if err != nil || calls.Load() != 3 || out.Metadata.Attempts != 3 {
			t.Fatalf("retry failed: %v", err)
		}
	})
	t.Run("respect-retry-after", func(t *testing.T) {
		var calls atomic.Int64
		c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			w.Header().Set("Retry-After", "3600")
			w.WriteHeader(429)
		})
		c.attempts = 3
		_, err := c.Rerank(context.Background(), rerankRequest())
		e := requireKind(t, err, RateLimit)
		if calls.Load() != 1 || e.Metadata.RetryAfter != time.Hour {
			t.Fatal("retried before Retry-After")
		}
	})
	t.Run("cancel-backoff", func(t *testing.T) {
		c := testClient(t, func(w http.ResponseWriter, r *http.Request) { w.Header().Set("Retry-After", "1"); w.WriteHeader(429) })
		c.attempts = 3
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
		defer cancel()
		_, err := c.Rerank(ctx, rerankRequest())
		requireKind(t, err, Timeout)
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatal("deadline cause lost")
		}
	})
	t.Run("pre-cancel", func(t *testing.T) {
		c, _ := NewClient(Config{APIKey: testKey})
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := c.Rerank(ctx, rerankRequest())
		requireKind(t, err, Cancelled)
		if !errors.Is(err, context.Canceled) {
			t.Fatal("cancel cause lost")
		}
	})
}

func TestRedirectDoesNotForwardCredential(t *testing.T) {
	var forwarded atomic.Int64
	dest := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { forwarded.Add(1) }))
	defer dest.Close()
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, dest.URL, 307) })
	_, err := c.Rerank(context.Background(), rerankRequest())
	requireKind(t, err, ProviderFailure)
	if forwarded.Load() != 0 {
		t.Fatal("redirect forwarded a paid/authenticated request")
	}
}

func TestTransportAndBodyErrors(t *testing.T) {
	c, _ := NewClient(Config{APIKey: testKey, MaxAttempts: 1, Transport: roundTrip(func(*http.Request) (*http.Response, error) { return nil, errors.New("secret=" + testKey) })})
	_, err := c.Rerank(context.Background(), rerankRequest())
	requireKind(t, err, TransportFailure)
	if strings.Contains(err.Error(), testKey) {
		t.Fatal("transport leaked key")
	}
	c.http.Transport = roundTrip(func(*http.Request) (*http.Response, error) { return nil, context.DeadlineExceeded })
	_, err = c.Rerank(context.Background(), rerankRequest())
	requireKind(t, err, Timeout)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("timeout cause lost")
	}
	for _, readErr := range []error{io.ErrUnexpectedEOF, context.DeadlineExceeded} {
		c.http.Transport = roundTrip(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: errorBody{readErr}}, nil
		})
		_, err = c.Rerank(context.Background(), rerankRequest())
		kind := TransportFailure
		if errors.Is(readErr, context.DeadlineExceeded) {
			kind = Timeout
		}
		requireKind(t, err, kind)
	}
	c = testClient(t, func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, goodRerank) })
	c.maxBytes = 10
	_, err = c.Rerank(context.Background(), rerankRequest())
	requireKind(t, err, ResponseTooLarge)
	var nilClient *Client
	_, err = nilClient.Rerank(context.Background(), rerankRequest())
	requireKind(t, err, InvalidConfig)
	_, err = c.Rerank(nil, rerankRequest())
	requireKind(t, err, InvalidConfig)
	_, err = c.post(context.Background(), "/rerank", make(chan int), &struct{}{})
	requireKind(t, err, InvalidInput)
}

type errorBody struct{ err error }

func (r errorBody) Read([]byte) (int, error) { return 0, r.err }
func (r errorBody) Close() error             { return nil }

func TestDiagnosticsHelpers(t *testing.T) {
	now := time.Date(2026, 9, 12, 10, 0, 0, 0, time.UTC)
	for value, want := range map[string]time.Duration{"": 0, "junk": 0, "0": 0, "-2": 0, " 12 ": 12 * time.Second, "999999999999999999": time.Duration(1<<63 - 1), "999999999999999999999999999999": time.Duration(1<<63 - 1), "18446744073709551615": time.Duration(1<<63 - 1), now.Add(2 * time.Second).Format(http.TimeFormat): 2 * time.Second, now.Add(-time.Hour).Format(http.TimeFormat): 0} {
		if got := retryAfter(value, now); got != want {
			t.Errorf("retry %s got %v want %v", value, got, want)
		}
	}
	for _, s := range []string{"", strings.Repeat("a", 129), testKey, "unsafe content", "bad\nheader"} {
		if safeToken(s, testKey) != "" {
			t.Error("unsafe token retained")
		}
	}
	if safeToken("req-1_A.b:c", testKey) != "req-1_A.b:c" {
		t.Fatal("valid token lost")
	}
	if err := wait(context.Background(), time.Nanosecond); err != nil {
		t.Fatal(err)
	}
}

func TestConcurrentMetadataIsPerRequest(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		var req RerankRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		w.Header().Set("X-Request-Id", req.Query)
		_, _ = io.WriteString(w, goodRerank)
	})
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			req := rerankRequest()
			req.Query = fmt.Sprintf("request-%d", i)
			out, err := c.Rerank(context.Background(), req)
			if err != nil {
				t.Error(err)
				return
			}
			if out.Metadata.RequestID != req.Query {
				t.Error("shared last response race")
			}
		}(i)
	}
	wg.Wait()
}

func TestOversizedRetryAfterNeverRetriesEarly(t *testing.T) {
	for _, v := range []string{"999999999999999999999999999999", "18446744073709551615"} {
		if retryAfter(v, time.Now()) != time.Duration(1<<63-1) {
			t.Fatal("retry-after overflow was ignored")
		}
	}
}
