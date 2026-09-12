package voyage

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

func intp(v int) *int { return &v }
func contextRequest() ContextRequest {
	return ContextRequest{Documents: []string{"full document A", "full document B"}, Model: "synthetic-context", Purpose: Document, Dimensions: 3, AutoChunk: true}
}

const goodContext = `{"data":[{"index":1,"data":[{"index":0,"text":"B passage","embedding":[1,2,3]}]},{"index":0,"data":[{"index":1,"text":"A second","embedding":[3,2,1]},{"index":0,"text":"A first","embedding":[1,1,1]}]}],"model":"synthetic-context","usage":{"total_tokens":123},"chunker_version":"1.0.0"}`

func TestContextAutoChunkContract(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/contextualizedembeddings" {
			t.Error("wrong endpoint")
		}
		var got struct {
			Inputs    []string `json:"inputs"`
			Model     string   `json:"model"`
			Purpose   Purpose  `json:"input_type"`
			Dimension int      `json:"output_dimension"`
			Auto      bool     `json:"enable_auto_chunking"`
			Size      int      `json:"chunk_size"`
			Overlap   int      `json:"chunk_overlap"`
			DType     string   `json:"output_dtype"`
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Inputs, contextRequest().Documents) || got.Purpose != Document || !got.Auto || got.Size != 512 || got.Overlap != 20 || got.Dimension != 3 || got.DType != "float" {
			t.Fatalf("contract %+v", got)
		}
		_, _ = io.WriteString(w, goodContext)
	})
	req := contextRequest()
	req.ChunkSize = intp(512)
	req.ChunkOverlap = intp(20)
	out, err := c.Contextualize(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Documents) != 2 || len(out.Documents[0].Chunks) != 2 || out.Documents[0].Chunks[0].Text != "A first" || out.Documents[0].Chunks[1].Text != "A second" || out.Documents[1].Chunks[0].Text != "B passage" {
		t.Fatalf("chunk identities %+v", out)
	}
	if out.Metadata.Usage.TotalTokens != 123 || out.Metadata.ChunkerVersion != "1.0.0" {
		t.Fatal("usage/version lost")
	}
}

func TestContextQueryAndGroupedInput(t *testing.T) {
	cases := []ContextRequest{
		{Documents: []string{"queryA", "queryB"}, Model: "synthetic-context", Purpose: Query, Dimensions: 3},
		{Chunks: [][]string{{"queryA"}, {"queryB"}}, Model: "synthetic-context", Purpose: Query, Dimensions: 3},
		{Chunks: [][]string{{"A first", "A second"}, {"B one"}}, Model: "synthetic-context", Purpose: Document, Dimensions: 3},
	}
	for i, req := range cases {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				var got map[string]json.RawMessage
				_ = json.NewDecoder(r.Body).Decode(&got)
				if _, ok := got["chunk_size"]; ok {
					t.Error("chunk knobs sent without auto")
				}
				if _, ok := got["chunk_overlap"]; ok {
					t.Error("overlap sent without auto")
				}
				if string(got["enable_auto_chunking"]) != "false" {
					t.Fatal("query/group auto chunked")
				}
				groups := req.Chunks
				if len(groups) == 0 {
					for _, s := range req.Documents {
						groups = append(groups, []string{s})
					}
				}
				rows := make([]any, len(groups))
				for di, g := range groups {
					chunks := make([]any, len(g))
					for ci := range g {
						chunks[ci] = map[string]any{"index": ci, "embedding": []float32{1, 2, 3}}
					}
					rows[di] = map[string]any{"index": di, "data": chunks}
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"data": rows})
			})
			out, err := c.Contextualize(context.Background(), req)
			if err != nil {
				t.Fatal(err)
			}
			want := "queryA"
			if req.Purpose == Document {
				want = "A first"
			}
			if out.Documents[0].Chunks[0].Text != want {
				t.Fatal("source text lost")
			}
		})
	}
}

func TestContext2048AndDistinctSpaceIdentity(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		var got map[string]any
		_ = json.NewDecoder(r.Body).Decode(&got)
		if got["output_dimension"] != float64(2048) || got["model"] != "voyage-context-4" {
			t.Fatal("2048 model option lost")
		}
		v := make([]float32, 2048)
		v[0] = 1
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"index": 0, "data": []any{map[string]any{"index": 0, "text": "supporting passage", "embedding": v}}}}, "model": "voyage-context-4"})
	})
	out, err := c.Contextualize(context.Background(), ContextRequest{Documents: []string{"full document"}, Model: "voyage-context-4", Purpose: Document, Dimensions: 2048, AutoChunk: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Documents[0].Chunks[0].Embedding) != 2048 {
		t.Fatal("dimension changed")
	}
	mm := c.EmbeddingSpace(MultimodalAPI, "voyage-multimodal-3.5", 2048)
	if out.Space.Key() == mm.Key() {
		t.Fatal("distinct model spaces collapsed")
	}
	other := out.Space
	other.Endpoint = "https://different.invalid/v1"
	if other.Key() == out.Space.Key() {
		t.Fatal("different endpoints collapsed")
	}
	var nilClient *Client
	if nilClient.EmbeddingSpace(ContextualizedAPI, "test", 3).Endpoint != "" {
		t.Fatal("nil client space")
	}
}

func TestContextInvalidInput(t *testing.T) {
	mutations := []func(*ContextRequest){
		func(r *ContextRequest) { r.Documents = nil }, func(r *ContextRequest) { r.Model = "" },
		func(r *ContextRequest) { r.Dimensions = 0 }, func(r *ContextRequest) { r.Dimensions = 65537 },
		func(r *ContextRequest) { r.Model = "voyage-context-4" },
		func(r *ContextRequest) { r.Purpose = "" }, func(r *ContextRequest) { r.Purpose = Query },
		func(r *ContextRequest) { r.Chunks = [][]string{{"chunk"}} },
		func(r *ContextRequest) { r.AutoChunk = false },
		func(r *ContextRequest) { r.Documents[0] = " " },
		func(r *ContextRequest) { r.Documents = make([]string, 1001) },
		func(r *ContextRequest) { r.ChunkSize = intp(0) }, func(r *ContextRequest) { r.ChunkSize = intp(32001) },
		func(r *ContextRequest) { r.ChunkOverlap = intp(-1) }, func(r *ContextRequest) { r.ChunkOverlap = intp(512) },
		func(r *ContextRequest) { r.AutoChunk = false; r.Purpose = Query; r.ChunkSize = intp(512) },
		func(r *ContextRequest) { r.AutoChunk = false; r.Purpose = Query; r.ChunkOverlap = intp(0) },
		func(r *ContextRequest) { r.AutoChunk = false; r.Documents = nil; r.Chunks = [][]string{{}} },
		func(r *ContextRequest) { r.AutoChunk = false; r.Documents = nil; r.Chunks = [][]string{{" "}} },
		func(r *ContextRequest) { r.AutoChunk = false; r.Documents = nil; r.Chunks = make([][]string, 1001) },
		func(r *ContextRequest) {
			r.AutoChunk = false
			r.Documents = nil
			r.Chunks = [][]string{make([]string, 16001)}
		},
		func(r *ContextRequest) {
			r.AutoChunk = false
			r.Documents = nil
			r.Purpose = Query
			r.Chunks = [][]string{{"a", "b"}}
		},
	}
	for i, mutate := range mutations {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			req := contextRequest()
			mutate(&req)
			_, _, err := validateContext(req)
			requireKind(t, err, InvalidInput)
		})
	}
}

func TestContextMalformedResponse(t *testing.T) {
	bodies := []string{
		`{}`, `null`, `{"data":[]}`,
		strings.Replace(goodContext, `"index":1,"data"`, `"index":0,"data"`, 1),
		strings.Replace(goodContext, `"index":1,"data"`, `"index":-1,"data"`, 1),
		strings.Replace(goodContext, `"index":1,"data"`, `"index":2,"data"`, 1),
		strings.Replace(goodContext, `"index":1,"data"`, `"data"`, 1),
		strings.Replace(goodContext, `"index":1,"text":"A second"`, `"index":0,"text":"A second"`, 1),
		strings.Replace(goodContext, `"index":1,"text":"A second"`, `"index":-1,"text":"A second"`, 1),
		strings.Replace(goodContext, `"index":1,"text":"A second"`, `"index":2,"text":"A second"`, 1),
		strings.Replace(goodContext, `"index":1,"text":"A second"`, `"text":"A second"`, 1),
		strings.Replace(goodContext, `"text":"B passage",`, ``, 1),
		strings.Replace(goodContext, `"text":"B passage"`, `"text":null`, 1),
		strings.Replace(goodContext, `"text":"B passage"`, `"text":" "`, 1),
		strings.Replace(goodContext, `[1,2,3]`, `[1,2]`, 1),
		strings.Replace(goodContext, `[1,2,3]`, `null`, 1),
		strings.Replace(goodContext, `[1,2,3]`, `[1,null,3]`, 1),
		strings.Replace(goodContext, `[1,2,3]`, `[0,0,0]`, 1),
		strings.Replace(goodContext, `[1,2,3]`, `[1,"2",3]`, 1),
		strings.Replace(goodContext, `[1,2,3]`, `[1,1e50,3]`, 1),
		strings.Replace(goodContext, `"embedding":[1,2,3]`, `"unrecognized":[1,2,3]`, 1),
		strings.Replace(goodContext, `"synthetic-context"`, `"wrong-space"`, 1),
	}
	for i, body := range bodies {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, body) })
			out, err := c.Contextualize(context.Background(), contextRequest())
			requireKind(t, err, MalformedResponse)
			if out != nil {
				t.Fatal("partial publication possible")
			}
		})
	}
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, goodContext) })
	req := contextRequest()
	req.AutoChunk = false
	req.Purpose = Query
	_, err := c.Contextualize(context.Background(), req)
	requireKind(t, err, MalformedResponse)
}

func TestDimensionsAndVectorValidation(t *testing.T) {
	for _, model := range []string{"voyage-context-4", "voyage-context-3", "voyage-multimodal-3.5"} {
		for _, d := range []int{256, 512, 1024, 2048} {
			if !validModelDimensions(model, d) {
				t.Fatal("supported dims rejected")
			}
		}
	}
	if !validModelDimensions("voyage-multimodal-3", 1024) || validModelDimensions("voyage-multimodal-3", 2048) {
		t.Fatal("legacy dims")
	}
	if validVector([]float32{float32(math.NaN())}, 1) || validVector([]float32{float32(math.Inf(1))}, 1) {
		t.Fatal("nonfinite accepted")
	}
	for _, usage := range []Usage{{TotalTokens: -1}, {TextTokens: -1}, {ImagePixels: -1}, {VideoPixels: -1}} {
		_, err := (envelope{Usage: &usage}).metadata(Metadata{}, "m")
		requireKind(t, err, MalformedResponse)
	}
}
