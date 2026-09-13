package voyage

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
)

const pngData = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

func mmRequest() MultimodalRequest {
	return MultimodalRequest{Model: "synthetic-multimodal", Purpose: Document, Dimensions: 3, Inputs: []MultimodalInput{{Content: []Part{{Type: "text", Text: "synthetic caption"}, {Type: "image_url", ImageURL: "https://images.example.invalid/figure.png?signature=synthetic"}}}, {Content: []Part{{Type: "text", Text: "text only"}}}}}
}

const goodMM = `{"data":[{"index":1,"embedding":[3,2,1]},{"index":0,"embedding":[1,2,3]}],"model":"synthetic-multimodal","usage":{"total_tokens":24,"text_tokens":8,"image_pixels":8960}}`

func TestMultimodalStructuredContract(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/multimodalembeddings" {
			t.Fatal("wrong endpoint")
		}
		var got MultimodalRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		if got.Purpose != Document || got.Dimensions != 3 || got.Truncation || got.Inputs[0].Content[1].Type != "image_url" || got.Inputs[0].Content[1].Text != "" || got.Inputs[0].Content[1].ImageURL != mmRequest().Inputs[0].Content[1].ImageURL {
			t.Fatalf("not a native image request %+v", got)
		}
		_, _ = io.WriteString(w, goodMM)
	})
	out, err := c.Multimodal(context.Background(), mmRequest())
	if err != nil {
		t.Fatal(err)
	}
	if out.Embeddings[0][0] != 1 || out.Embeddings[1][0] != 3 || out.Metadata.Usage.ImagePixels != 8960 {
		t.Fatal("index/usage lost")
	}
}

func TestMultimodal2048QueriesAndBase64(t *testing.T) {
	for _, image := range []bool{false, true} {
		t.Run(fmt.Sprint(image), func(t *testing.T) {
			req := MultimodalRequest{Model: "voyage-multimodal-3.5", Purpose: Query, Dimensions: 2048, Truncation: true, Inputs: []MultimodalInput{{Content: []Part{{Type: "text", Text: "find diagrams"}}}}}
			if image {
				req.Inputs[0].Content = append(req.Inputs[0].Content, Part{Type: "image_base64", ImageBase64: pngData})
			}
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				var got MultimodalRequest
				_ = json.NewDecoder(r.Body).Decode(&got)
				if got.Purpose != Query || got.Dimensions != 2048 || !got.Truncation {
					t.Fatal("query options dropped")
				}
				if image && got.Inputs[0].Content[1].ImageBase64 != pngData {
					t.Fatal("base64 altered")
				}
				v := make([]float32, 2048)
				v[0] = 1
				_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"index": 0, "embedding": v}}, "model": req.Model})
			})
			out, err := c.Multimodal(context.Background(), req)
			if err != nil || len(out.Embeddings[0]) != 2048 {
				t.Fatalf("2048 query: %v", err)
			}
		})
	}
}

func TestMultimodalInputValidation(t *testing.T) {
	mutations := []func(*MultimodalRequest){
		func(r *MultimodalRequest) { r.Model = "" }, func(r *MultimodalRequest) { r.Purpose = "" },
		func(r *MultimodalRequest) { r.Dimensions = -1 }, func(r *MultimodalRequest) { r.Inputs = nil },
		func(r *MultimodalRequest) { r.Inputs = make([]MultimodalInput, 1001) },
		func(r *MultimodalRequest) { r.Inputs[0].Content = nil },
		func(r *MultimodalRequest) { r.Inputs[0].Content[0].Text = " " },
		func(r *MultimodalRequest) { r.Inputs[0].Content[0].ImageURL = "https://example.invalid/image" },
		func(r *MultimodalRequest) { r.Inputs[0].Content[0].ImageBase64 = pngData },
		func(r *MultimodalRequest) { r.Inputs[0].Content[1].Text = "ambiguous" },
		func(r *MultimodalRequest) { r.Inputs[0].Content[1].ImageBase64 = pngData },
		func(r *MultimodalRequest) { r.Inputs[0].Content[1].ImageURL = "file:///etc/passwd" },
		func(r *MultimodalRequest) { r.Inputs[0].Content[1].ImageURL = ":bad" },
		func(r *MultimodalRequest) { r.Inputs[0].Content[1].ImageURL = "https:///missing" },
		func(r *MultimodalRequest) {
			r.Inputs[0].Content[1].ImageURL = "https://user:password@example.invalid/image"
		},
		func(r *MultimodalRequest) { r.Inputs[0].Content[1].ImageURL = "https://example.invalid/image#fragment" },
		func(r *MultimodalRequest) { r.Inputs[0].Content[1].Type = "video_url" },
		func(r *MultimodalRequest) {
			r.Inputs[1].Content = append(r.Inputs[1].Content, Part{Type: "image_base64", ImageBase64: pngData})
		},
		func(r *MultimodalRequest) {
			r.Inputs[0].Content = []Part{{Type: "image_base64", Text: "ambiguous", ImageBase64: pngData}}
		},
		func(r *MultimodalRequest) {
			r.Inputs[0].Content = []Part{{Type: "image_base64", ImageURL: "https://example.invalid/image", ImageBase64: pngData}}
		},
		func(r *MultimodalRequest) {
			r.Inputs[0].Content = []Part{{Type: "image_base64", ImageBase64: "broken"}}
		},
	}
	for i, mutate := range mutations {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			req := mmRequest()
			mutate(&req)
			requireKind(t, ValidateMultimodal(req), InvalidInput)
		})
	}
	for _, data := range []string{"", "data:image/png;base64,", "data:image/svg+xml;base64,YQ==", "data:image/png;base64,????", "data:image/png;base64,YR==", "data:image/png;base64," + strings.Repeat("A", base64.StdEncoding.EncodedLen(20_000_000)+1)} {
		if validImageDataURL(data) {
			t.Error("invalid base64 accepted")
		}
	}
	for _, mime := range []string{"png", "jpeg", "webp", "gif"} {
		if !validImageDataURL("data:image/" + mime + ";base64,YQ==") {
			t.Fatal("allowed MIME rejected")
		}
	}
}

func TestMultimodalMalformedResponse(t *testing.T) {
	bodies := []string{
		`{}`, `{"data":[]}`, `{"data":[{"index":0,"embedding":[1,2,3]}]}`,
		strings.Replace(goodMM, `"index":1`, `"index":0`, 1),
		strings.Replace(goodMM, `"index":1`, `"index":-1`, 1),
		strings.Replace(goodMM, `"index":1`, `"index":2`, 1),
		strings.Replace(goodMM, `"index":1,`, ``, 1),
		strings.Replace(goodMM, `[3,2,1]`, `[3,2]`, 1),
		strings.Replace(goodMM, `[3,2,1]`, `[3,null,1]`, 1),
		strings.Replace(goodMM, `[3,2,1]`, `[0,0,0]`, 1),
		strings.Replace(goodMM, `"synthetic-multimodal"`, `"wrong-model"`, 1),
	}
	for i, body := range bodies {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			c := testClient(t, func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, body) })
			out, err := c.Multimodal(context.Background(), mmRequest())
			requireKind(t, err, MalformedResponse)
			if out != nil {
				t.Fatal("malformed partial success")
			}
		})
	}
}

func TestEmbeddingHTTPFailurePropagation(t *testing.T) {
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(401) })
	_, err := c.Contextualize(context.Background(), contextRequest())
	requireKind(t, err, Authentication)
	_, err = c.Multimodal(context.Background(), mmRequest())
	requireKind(t, err, Authentication)
	req := contextRequest()
	req.Purpose = "invalid"
	_, err = c.Contextualize(context.Background(), req)
	requireKind(t, err, InvalidInput)
	mm := mmRequest()
	mm.Purpose = "invalid"
	_, err = c.Multimodal(context.Background(), mm)
	requireKind(t, err, InvalidInput)
}
