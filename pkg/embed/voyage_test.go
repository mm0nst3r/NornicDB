package embed

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/orneryd/nornicdb/pkg/voyage"
)

func voyageBool(v bool) *bool { return &v }
func voyageInt(v int) *int    { return &v }
func nativeEmbedder(t *testing.T, mode string, handler http.HandlerFunc, options *VoyageEmbeddingOptions) *VoyageEmbedder {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	if options == nil {
		options = &VoyageEmbeddingOptions{Mode: mode, MaxAttempts: 1}
	}
	e, err := NewVoyage(&Config{APIURL: server.URL, APIKey: "synthetic-key", Model: "synthetic-model", Dimensions: 3}, options)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestVoyageManagedDocumentPreservesChunksAndPurpose(t *testing.T) {
	var calls atomic.Int64
	e := nativeEmbedder(t, "contextualized", func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["input_type"] != "document" || body["enable_auto_chunking"] != true || body["inputs"].([]any)[0] != "the ENTIRE original document" {
			t.Error("pre-split or wrong-purpose document")
		}
		_, _ = io.WriteString(w, `{"data":[{"index":0,"data":[{"index":1,"text":"Second passage","embedding":[2,1,1]},{"index":0,"text":"First passage","embedding":[1,2,1]}]}],"usage":{"total_tokens":12},"chunker_version":"1.0.0"}`)
	}, nil)
	in, err := e.PrepareDocument("the ENTIRE original document", nil)
	if err != nil {
		t.Fatal(err)
	}
	out, err := e.EmbedDocument(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Embeddings) != 2 || out.ChunkTexts[0] != "First passage" || out.ChunkTexts[1] != "Second passage" || out.InputFingerprint != in.Fingerprint() || out.Metadata.Usage.TotalTokens != 12 || calls.Load() != 1 {
		t.Fatalf("lost chunks %+v", out)
	}
	if e.Model() != "synthetic-model" || e.Dimensions() != 3 || e.Backend() != "cpu" || out.Space.Key() != e.EmbeddingSpace().Key() {
		t.Fatal("provider identity")
	}
}

func TestVoyageExplicitQueryBypassesLegacyChunkingAndCaching(t *testing.T) {
	var calls atomic.Int64
	e := nativeEmbedder(t, "contextualized", func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["input_type"] != "query" || body["enable_auto_chunking"] != false || body["chunk_size"] != nil {
			t.Error("query auto-chunked")
		}
		_, _ = io.WriteString(w, `{"data":[{"index":0,"data":[{"index":0,"embedding":[1,2,3]}]}]}`)
	}, nil)
	if _, err := e.Embed(context.Background(), "query"); !errors.Is(err, ErrExplicitEmbeddingPurpose) {
		t.Fatal("ambiguous purpose accepted")
	}
	if _, err := e.EmbedBatch(context.Background(), []string{"doc"}); !errors.Is(err, ErrExplicitEmbeddingPurpose) {
		t.Fatal("managed context flattened")
	}
	chunks, err := e.ChunkText("whole input", 1, 0)
	if err != nil || len(chunks) != 1 || chunks[0] != "whole input" {
		t.Fatal("legacy splitting")
	}
	wrapped := NewCachedEmbedder(NewCachedEmbedder(e, 100), 100)
	p, ok := ManagedProvider(wrapped)
	if !ok || p != e {
		t.Fatal("cache lost managed capability")
	}
	for i := 0; i < 2; i++ {
		v, err := QueryVector(context.Background(), wrapped, "query")
		if err != nil || len(v) != 3 {
			t.Fatal(err)
		}
	}
	if calls.Load() != 2 {
		t.Fatal("native query entered legacy text-only cache")
	}
	if _, err := e.EmbedMultimodalQuery(context.Background(), nil); err == nil {
		t.Fatal("query model space crossed")
	}
}

func TestVoyageGroupedDocumentAndMultimodal(t *testing.T) {
	e := nativeEmbedder(t, "contextualized", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["enable_auto_chunking"] != false || body["inputs"].([]any)[0].([]any)[0] != "whole" {
			t.Error("grouped document contract")
		}
		_, _ = io.WriteString(w, `{"data":[{"index":0,"data":[{"index":0,"embedding":[1,2,3]}]}]}`)
	}, &VoyageEmbeddingOptions{Mode: "contextualized", AutoChunking: voyageBool(false), MaxAttempts: 1})
	out, err := e.EmbedDocument(context.Background(), ManagedInput{Text: "whole"})
	if err != nil || out.ChunkTexts[0] != "whole" {
		t.Fatal("grouped source text lost")
	}
	mm := nativeEmbedder(t, "multimodal", func(w http.ResponseWriter, r *http.Request) {
		var body voyage.MultimodalRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		if r.URL.Path != "/v1/multimodalembeddings" || body.Dimensions != 3 || body.Truncation {
			t.Error("mm native contract")
		}
		if len(body.Inputs[0].Content) == 2 && body.Inputs[0].Content[1].Type != "image_url" {
			t.Error("URL embedded as text")
		}
		_, _ = io.WriteString(w, `{"data":[{"index":0,"embedding":[1,2,3]}]}`)
	}, nil)
	properties := map[string]any{"_embedding_content": []map[string]any{{"type": "text", "text": "caption"}, {"type": "image_url", "image_url": "https://example.invalid/image.png"}}}
	input, err := mm.PrepareDocument("unrelated flattened text", properties)
	if err != nil {
		t.Fatal(err)
	}
	if input.Text != "" || len(input.Content) != 2 || input.Content[0].Text != "caption" {
		t.Fatal("structured content was flattened or augmented")
	}
	out, err = mm.EmbedDocument(context.Background(), input)
	if err != nil || len(out.Embeddings) != 1 || out.ChunkTexts[0] != "caption" {
		t.Fatal(err)
	}
	if e.EmbeddingSpace().Key() == mm.EmbeddingSpace().Key() {
		t.Fatal("equal dimensions confused model spaces")
	}
	if _, err := mm.EmbedQuery(context.Background(), "find caption"); err != nil {
		t.Fatal(err)
	}
	if _, err := mm.EmbedMultimodalQuery(context.Background(), input.Content); err != nil {
		t.Fatal(err)
	}
	textOnly, err := mm.PrepareDocument("only text", nil)
	if err != nil || textOnly.Content[0].Text != "only text" {
		t.Fatal("text-only multimodal fallback")
	}
}

func TestVoyageConfigurationValidation(t *testing.T) {
	if _, err := NewVoyage(nil, nil); err == nil {
		t.Fatal("no key accepted")
	}
	base := Config{APIKey: "synthetic-key"}
	for _, provider := range []string{"voyage-context", "voyage-multimodal"} {
		cfg := base
		cfg.Provider = provider
		e, err := NewVoyage(&cfg, nil)
		if err != nil || e.Dimensions() != 2048 {
			t.Fatal("defaults", err)
		}
	}
	badConfigs := []Config{{APIKey: "k", Provider: "unknown"}, {APIKey: "k", Dimensions: -1}, {APIKey: "k", APIPath: "/v1/embeddings"}, {APIKey: "k", Dimensions: 3}, {APIKey: "k", APIURL: "http://insecure.example.invalid"}}
	for _, cfg := range badConfigs {
		if _, err := NewVoyage(&cfg, nil); err == nil {
			t.Fatal("invalid config accepted")
		}
	}
	badOptions := []VoyageEmbeddingOptions{{Mode: "unknown"}, {ContentProperty: " "}, {Mode: "multimodal", AutoChunking: voyageBool(true)}, {Mode: "multimodal", ChunkSize: voyageInt(512)}, {AutoChunking: voyageBool(false), ChunkOverlap: voyageInt(0)}, {ChunkSize: voyageInt(0)}, {ChunkSize: voyageInt(32001)}, {ChunkOverlap: voyageInt(-1)}, {ChunkOverlap: voyageInt(512)}, {MaxAttempts: 11}}
	for _, opts := range badOptions {
		if _, err := NewVoyage(&base, &opts); err == nil {
			t.Fatal("invalid options accepted")
		}
	}
	size, overlap := 600, 25
	cfg := base
	cfg.Model = "synthetic"
	cfg.Dimensions = 3
	e, err := NewVoyage(&cfg, &VoyageEmbeddingOptions{ChunkSize: &size, ChunkOverlap: &overlap})
	if err != nil {
		t.Fatal(err)
	}
	size = 1
	overlap = 100
	cfg.Model = "mutated"
	if *e.options.ChunkSize != 600 || *e.options.ChunkOverlap != 25 || e.Model() != "synthetic" {
		t.Fatal("mutable configuration retained")
	}
}

func TestVoyageDocumentValidationAndFailures(t *testing.T) {
	e := nativeEmbedder(t, "contextualized", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(401) }, nil)
	if _, err := e.PrepareDocument("", nil); err == nil {
		t.Fatal("empty doc")
	}
	if _, err := e.PrepareDocument("text", map[string]any{"_embedding_content": "url"}); err == nil {
		t.Fatal("multimodal silently used contextual space")
	}
	if _, err := e.EmbedDocument(context.Background(), ManagedInput{Content: []voyage.Part{{Type: "text", Text: "wrong"}}}); err == nil {
		t.Fatal("wrong input type")
	}
	if _, err := e.EmbedDocument(context.Background(), ManagedInput{Text: "text"}); err == nil {
		t.Fatal("HTTP failure swallowed")
	}
	if _, err := e.EmbedQuery(context.Background(), "text"); err == nil {
		t.Fatal("query HTTP failure swallowed")
	}
	mm := nativeEmbedder(t, "multimodal", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(429) }, nil)
	for i, raw := range []any{"bare URL", nil, make(chan int), []map[string]any{{"type": "text", "text": "ok", "unknown": "drop me"}}, []map[string]any{{"type": "image_url", "image_url": "file:///etc/passwd"}}} {
		if _, err := mm.PrepareDocument("text", map[string]any{"_embedding_content": raw}); err == nil {
			t.Fatalf("invalid structure %d accepted", i)
		}
	}
	if _, err := mm.EmbedDocument(context.Background(), ManagedInput{Text: "flat input"}); err == nil {
		t.Fatal("flat mm input")
	}
	if _, err := mm.EmbedDocument(context.Background(), ManagedInput{Content: []voyage.Part{{Type: "text", Text: "text"}}}); err == nil {
		t.Fatal("mm HTTP failure swallowed")
	}
	if _, err := mm.EmbedQuery(context.Background(), "text"); err == nil {
		t.Fatal("mm query HTTP failure swallowed")
	}
}

type legacyVoyageFixture struct{ calls int }

func (l *legacyVoyageFixture) Embed(context.Context, string) ([]float32, error) {
	l.calls++
	return []float32{1, 2, 3}, nil
}
func (l *legacyVoyageFixture) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, fmt.Errorf("unused")
}
func (l *legacyVoyageFixture) ChunkText(s string, _, _ int) ([]string, error) {
	return []string{s}, nil
}
func (l *legacyVoyageFixture) Dimensions() int { return 3 }
func (l *legacyVoyageFixture) Model() string   { return "legacy" }
func (l *legacyVoyageFixture) Backend() string { return "cpu" }

func TestVoyageCapabilityFallbackAndFingerprint(t *testing.T) {
	legacy := &legacyVoyageFixture{}
	if _, ok := ManagedProvider(legacy); ok {
		t.Fatal("legacy falsely managed")
	}
	var nilCached *CachedEmbedder
	if _, ok := ManagedProvider(nilCached); ok {
		t.Fatal("nil cache falsely managed")
	}
	if _, err := QueryVector(context.Background(), nil, "text"); err == nil {
		t.Fatal("nil query embedder")
	}
	if _, err := nilCached.EmbedQuery(context.Background(), "text"); err == nil {
		t.Fatal("nil cache query")
	}
	if _, err := QueryVector(context.Background(), legacy, "text"); err != nil {
		t.Fatal(err)
	}
	if _, err := NewCachedEmbedder(legacy, 10).EmbedQuery(context.Background(), "text"); err != nil {
		t.Fatal(err)
	}
	if legacy.calls != 2 {
		t.Fatal("legacy delegation changed")
	}
	a, b := ManagedInput{Text: "a"}, ManagedInput{Text: "b"}
	if a.Fingerprint() == b.Fingerprint() || strings.Contains(a.Fingerprint(), "secret text") || a.Fingerprint() != a.Fingerprint() {
		t.Fatal("fingerprint")
	}
}
