package voyage

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"strings"
)

// Purpose distinguishes compatible query and document prompts in the same model
// space. An empty/unknown purpose is rejected so callers cannot accidentally omit it.
type Purpose string

const (
	Query             Purpose = "query"
	Document          Purpose = "document"
	ContextualizedAPI         = "contextualizedembeddings"
	MultimodalAPI             = "multimodalembeddings"
)

// Space is the identity of an embedding space. Query/document purpose is NOT a
// component because those prompts intentionally share a space. API family, model,
// dimensions, endpoint, and dtype are components. A Space must be enforced by the
// index owner; recording it alone does not make an unpartitioned index safe.
type Space struct {
	API        string `json:"api"`
	Model      string `json:"model"`
	Dimensions int    `json:"dimensions"`
	Endpoint   string `json:"endpoint"`
	DType      string `json:"dtype"`
}

// Key is a stable, credential-free identifier for storage/index partition checks.
func (s Space) Key() string {
	b, _ := json.Marshal(s)
	h := sha256.Sum256(b)
	return "voyage:" + hex.EncodeToString(h[:])
}

// EmbeddingSpace returns a descriptor for this endpoint, API family and model.
func (c *Client) EmbeddingSpace(api, model string, dimensions int) Space {
	endpoint := ""
	if c != nil {
		endpoint = c.baseURL
	}
	return Space{API: api, Model: model, Dimensions: dimensions, Endpoint: endpoint, DType: "float32"}
}

// ContextRequest preserves whole-document context. Use Documents with AutoChunk
// for full documents; use Chunks for already chunked document groups. Query inputs
// use Documents (one query per item) or Chunks with exactly one string per group.
// ChunkSize/Overlap are pointers so omitted and explicit zero remain distinct.
type ContextRequest struct {
	Documents    []string
	Chunks       [][]string
	Model        string
	Purpose      Purpose
	Dimensions   int
	AutoChunk    bool
	ChunkSize    *int
	ChunkOverlap *int
}

// Chunk retains both the provider-generated passage and its embedding. Index is
// the chunk's zero-based position within its original document, not response order.
type Chunk struct {
	Index     int       `json:"index"`
	Text      string    `json:"text"`
	Embedding []float32 `json:"embedding"`
}

// ContextDocument is one original document/query with chunks in original order.
type ContextDocument struct {
	Index  int     `json:"index"`
	Chunks []Chunk `json:"chunks"`
}

// ContextResponse preserves original document/chunk identity and diagnostics.
type ContextResponse struct {
	Documents []ContextDocument `json:"documents"`
	Space     Space             `json:"space"`
	Metadata  Metadata          `json:"metadata"`
}

// Contextualize sends each complete document/group in a single request. It never
// independently embeds constituent chunks or silently reduces output dimensions.
// Auto-chunked responses must include the actual returned passage text.
func (c *Client) Contextualize(ctx context.Context, req ContextRequest) (*ContextResponse, error) {
	inputs, counts, err := validateContext(req)
	if err != nil {
		return nil, err
	}
	payload := struct {
		Inputs     any     `json:"inputs"`
		Model      string  `json:"model"`
		Purpose    Purpose `json:"input_type"`
		Dimensions int     `json:"output_dimension"`
		DType      string  `json:"output_dtype"`
		Auto       bool    `json:"enable_auto_chunking"`
		Size       *int    `json:"chunk_size,omitempty"`
		Overlap    *int    `json:"chunk_overlap,omitempty"`
	}{inputs, req.Model, req.Purpose, req.Dimensions, "float", req.AutoChunk, req.ChunkSize, req.ChunkOverlap}
	var wire struct {
		envelope
		Data []struct {
			Index *int `json:"index"`
			Data  []struct {
				Index     *int       `json:"index"`
				Text      *string    `json:"text"`
				Embedding vectorWire `json:"embedding"`
			} `json:"data"`
		} `json:"data"`
	}
	meta, err := c.post(ctx, "/contextualizedembeddings", payload, &wire)
	if err != nil {
		return nil, err
	}
	meta, err = wire.envelope.metadata(meta, req.Model)
	if err != nil {
		return nil, err
	}
	if len(wire.Data) != len(counts) {
		return nil, failure(MalformedResponse, meta)
	}
	out := &ContextResponse{Documents: make([]ContextDocument, len(counts)),
		Space: c.EmbeddingSpace(ContextualizedAPI, req.Model, req.Dimensions), Metadata: meta}
	seen := make([]bool, len(counts))
	total := 0
	for _, doc := range wire.Data {
		if doc.Index == nil || *doc.Index < 0 || *doc.Index >= len(counts) || seen[*doc.Index] || len(doc.Data) == 0 {
			return nil, failure(MalformedResponse, meta)
		}
		di := *doc.Index
		seen[di] = true
		if counts[di] > 0 && len(doc.Data) != counts[di] {
			return nil, failure(MalformedResponse, meta)
		}
		total += len(doc.Data)
		if total > 16000 {
			return nil, failure(MalformedResponse, meta)
		}
		cd := ContextDocument{Index: di, Chunks: make([]Chunk, len(doc.Data))}
		cs := make([]bool, len(doc.Data))
		for _, row := range doc.Data {
			if row.Index == nil || *row.Index < 0 || *row.Index >= len(cs) || cs[*row.Index] ||
				!validVector(row.Embedding, req.Dimensions) {
				return nil, failure(MalformedResponse, meta)
			}
			ci := *row.Index
			cs[ci] = true
			var text string
			if req.AutoChunk {
				if row.Text == nil || strings.TrimSpace(*row.Text) == "" {
					return nil, failure(MalformedResponse, meta)
				}
				text = *row.Text
			} else if len(req.Chunks) > 0 {
				text = req.Chunks[di][ci]
			} else {
				text = req.Documents[di]
			}
			cd.Chunks[ci] = Chunk{Index: ci, Text: text, Embedding: []float32(row.Embedding)}
		}
		out.Documents[di] = cd
	}
	return out, nil
}

func validateContext(req ContextRequest) (any, []int, error) {
	invalid := func() (any, []int, error) { return nil, nil, failure(InvalidInput, Metadata{}) }
	if !validModelDimensions(req.Model, req.Dimensions) || (req.Purpose != Query && req.Purpose != Document) ||
		(len(req.Documents) == 0) == (len(req.Chunks) == 0) {
		return invalid()
	}
	if !req.AutoChunk && (req.ChunkSize != nil || req.ChunkOverlap != nil) {
		return invalid()
	}
	if req.AutoChunk && (req.Purpose != Document || len(req.Chunks) != 0) {
		return invalid()
	}
	if req.Purpose == Document && !req.AutoChunk && len(req.Chunks) == 0 {
		return invalid()
	}
	if req.AutoChunk {
		size := 512
		if req.ChunkSize != nil {
			size = *req.ChunkSize
		}
		if size < 1 || size > 32000 {
			return invalid()
		}
		if req.ChunkOverlap != nil && (*req.ChunkOverlap < 0 || *req.ChunkOverlap >= size) {
			return invalid()
		}
	}
	if len(req.Documents) > 0 {
		if len(req.Documents) > 1000 {
			return invalid()
		}
		counts := make([]int, len(req.Documents))
		for i, text := range req.Documents {
			if strings.TrimSpace(text) == "" {
				return invalid()
			}
			if !req.AutoChunk {
				counts[i] = 1
			}
		}
		return req.Documents, counts, nil
	}
	if len(req.Chunks) > 1000 {
		return invalid()
	}
	counts := make([]int, len(req.Chunks))
	total := 0
	for i, group := range req.Chunks {
		if len(group) == 0 || (req.Purpose == Query && len(group) != 1) {
			return invalid()
		}
		counts[i] = len(group)
		total += len(group)
		if total > 16000 {
			return invalid()
		}
		for _, text := range group {
			if strings.TrimSpace(text) == "" {
				return invalid()
			}
		}
	}
	return req.Chunks, counts, nil
}

func validModelDimensions(model string, dimensions int) bool {
	if strings.TrimSpace(model) == "" || dimensions <= 0 || dimensions > 65536 {
		return false
	}
	// Known model dimensions are validated locally; future model names remain
	// configurable and are validated by the provider. 65536 is a client resource cap.
	switch model {
	case "voyage-context-4", "voyage-context-3", "voyage-multimodal-3.5":
		return dimensions == 256 || dimensions == 512 || dimensions == 1024 || dimensions == 2048
	case "voyage-multimodal-3":
		return dimensions == 1024
	default:
		return true
	}
}

// vectorWire rejects JSON null both as a vector and inside a vector. encoding/json
// otherwise silently turns null array elements into zero-valued float32s.
type vectorWire []float32

func (v *vectorWire) UnmarshalJSON(data []byte) error {
	if bytes.Contains(data, []byte("null")) {
		return failure(MalformedResponse, Metadata{})
	}
	var values []float32
	if err := json.Unmarshal(data, &values); err != nil {
		return failure(MalformedResponse, Metadata{})
	}
	*v = values
	return nil
}

func validVector(values []float32, dimensions int) bool {
	if len(values) != dimensions {
		return false
	}
	nonzero := false
	for _, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
		if v != 0 {
			nonzero = true
		}
	}
	return nonzero
}
