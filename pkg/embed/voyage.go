package embed

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/orneryd/nornicdb/pkg/voyage"
)

// ErrExplicitEmbeddingPurpose prevents a native managed provider from silently
// treating document text as a query or discarding all but its first chunk.
// Use EmbedQuery for a query, and EmbedDocument for a managed document.
var ErrExplicitEmbeddingPurpose = errors.New("voyage embedding requires explicit query or managed-document purpose")

// VoyageEmbeddingOptions configures the two distinct native embedding spaces.
// Mode is "contextualized" or "multimodal"; it can also be selected with Config's
// Provider="voyage-context" / "voyage-multimodal". AutoChunking defaults to true
// for contextualized documents. ContentProperty is an explicitly structured input
// array (default "_embedding_content"), never a bare image URL string.
type VoyageEmbeddingOptions struct {
	Mode            string
	AutoChunking    *bool
	ChunkSize       *int
	ChunkOverlap    *int
	Truncation      bool
	ContentProperty string
	MaxAttempts     int
	MaxRetryDelay   time.Duration
}

// ManagedInput is the complete input to one managed document operation. Text is
// used for contextualized input; Content is interleaved multimodal input. Its
// fingerprint supports a stale-input check before the existing worker publishes.
type ManagedInput struct {
	Text    string        `json:"text,omitempty"`
	Content []voyage.Part `json:"content,omitempty"`
}

// Fingerprint is a deterministic, content-free diagnostic identity, not a cache
// key shared across models. Model-space identity is carried separately.
func (in ManagedInput) Fingerprint() string {
	b, _ := json.Marshal(in)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// ManagedDocumentResult retains all chunks and request diagnostics together. The
// worker (not this provider) owns persistence, retries, and index publication.
type ManagedDocumentResult struct {
	Embeddings       [][]float32
	ChunkTexts       []string
	Space            voyage.Space
	Metadata         voyage.Metadata
	InputFingerprint string
}

// ManagedDocumentEmbedder is an optional worker capability. It accepts a complete
// document and returns all chunks in one result, bypassing legacy micro-batching.
type ManagedDocumentEmbedder interface {
	ContentProperty() string
	PrepareDocument(text string, properties map[string]any) (ManagedInput, error)
	EmbedDocument(context.Context, ManagedInput) (ManagedDocumentResult, error)
	EmbeddingSpace() voyage.Space
}

// VoyageEmbedder is a native contextualized or multimodal provider. Legacy Embed
// and EmbedBatch intentionally error rather than guess query/document purpose.
// The managed worker capability and explicit query capability are separate.
type VoyageEmbedder struct {
	client    *voyage.Client
	config    Config
	options   VoyageEmbeddingOptions
	autoChunk bool
	api       string
}

var _ Embedder = (*VoyageEmbedder)(nil)
var _ ManagedDocumentEmbedder = (*VoyageEmbedder)(nil)

// NewVoyage uses existing Config credential/endpoint/model/dimension fields.
// Defaults are voyage-context-4 or voyage-multimodal-3.5, with 2048 dimensions.
// It validates native options without generating embeddings or accessing images.
//
// Example:
//
//	e, err := embed.NewVoyage(&embed.Config{
//	    Provider: "voyage-context", APIKey: os.Getenv("VOYAGE_API_KEY"), Dimensions: 2048,
//	}, nil)
//	if err != nil { return err }
//	queryVector, err := e.EmbedQuery(ctx, "Which passage supports this claim?")
func NewVoyage(config *Config, options *VoyageEmbeddingOptions) (*VoyageEmbedder, error) {
	cfg := Config{}
	if config != nil {
		cfg = *config
	}
	opts := VoyageEmbeddingOptions{}
	if options != nil {
		opts = *options
	}
	// The managed worker owns durable retry scheduling. A direct provider call
	// makes one attempt by default; explicit clients may configure otherwise.
	if opts.MaxAttempts == 0 {
		opts.MaxAttempts = 1
	}
	if opts.Mode == "" {
		switch cfg.Provider {
		case "voyage-context", "":
			opts.Mode = "contextualized"
		case "voyage-multimodal":
			opts.Mode = "multimodal"
		default:
			return nil, fmt.Errorf("voyage embedding: invalid provider mode")
		}
	}
	api := voyage.ContextualizedAPI
	switch opts.Mode {
	case "contextualized":
		if cfg.Model == "" {
			cfg.Model = "voyage-context-4"
		}
	case "multimodal":
		api = voyage.MultimodalAPI
		if cfg.Model == "" {
			cfg.Model = "voyage-multimodal-3.5"
		}
	default:
		return nil, fmt.Errorf("voyage embedding: invalid provider mode")
	}
	if cfg.Dimensions == 0 {
		cfg.Dimensions = 2048
	}
	if cfg.Dimensions < 1 {
		return nil, fmt.Errorf("voyage embedding: invalid dimensions")
	}
	if cfg.APIURL == "" {
		cfg.APIURL = "https://api.voyageai.com"
	}
	if cfg.APIPath == "" {
		cfg.APIPath = "/v1/" + api
	}
	endpoint := strings.TrimRight(cfg.APIURL, "/") + cfg.APIPath
	if !strings.HasSuffix(endpoint, "/"+api) {
		return nil, fmt.Errorf("voyage embedding: endpoint does not match provider mode")
	}
	if opts.ContentProperty == "" {
		opts.ContentProperty = "_embedding_content"
	}
	if strings.TrimSpace(opts.ContentProperty) == "" {
		return nil, fmt.Errorf("voyage embedding: content property is required")
	}
	auto := true
	if opts.AutoChunking != nil {
		auto = *opts.AutoChunking
	}
	if opts.Mode == "multimodal" && (opts.AutoChunking != nil || opts.ChunkSize != nil || opts.ChunkOverlap != nil) {
		return nil, fmt.Errorf("voyage embedding: contextual chunk options are invalid for multimodal input")
	}
	if !auto && (opts.ChunkSize != nil || opts.ChunkOverlap != nil) {
		return nil, fmt.Errorf("voyage embedding: chunk options require automatic chunking")
	}
	size := 512
	if opts.ChunkSize != nil {
		size = *opts.ChunkSize
		v := size
		opts.ChunkSize = &v
	}
	if opts.ChunkOverlap != nil {
		v := *opts.ChunkOverlap
		opts.ChunkOverlap = &v
	}
	if size < 1 || size > 32000 || opts.ChunkOverlap != nil && (*opts.ChunkOverlap < 0 || *opts.ChunkOverlap >= size) {
		return nil, fmt.Errorf("voyage embedding: invalid chunk size or overlap")
	}
	// Validate known model dimensions locally via the public request validator.
	if err := voyage.ValidateMultimodal(voyage.MultimodalRequest{Inputs: []voyage.MultimodalInput{{Content: []voyage.Part{{Type: "text", Text: "validate"}}}}, Model: cfg.Model, Purpose: voyage.Query, Dimensions: cfg.Dimensions}); err != nil {
		return nil, err
	}
	client, err := voyage.NewClient(voyage.Config{APIKey: cfg.APIKey, BaseURL: strings.TrimSuffix(endpoint, "/"+api), Timeout: cfg.Timeout, MaxAttempts: opts.MaxAttempts, MaxRetryDelay: opts.MaxRetryDelay})
	if err != nil {
		return nil, err
	}
	return &VoyageEmbedder{client: client, config: cfg, options: opts, autoChunk: auto, api: api}, nil
}

// Embed refuses to infer input purpose or silently collapse managed chunks.
func (e *VoyageEmbedder) Embed(context.Context, string) ([]float32, error) {
	return nil, ErrExplicitEmbeddingPurpose
}

// EmbedBatch refuses to flatten document context or mix query/document cache keys.
func (e *VoyageEmbedder) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, ErrExplicitEmbeddingPurpose
}

// ChunkText returns the whole input; native contextualized chunking belongs to
// Voyage, and query embedding must not be silently truncated by legacy chunk caps.
func (e *VoyageEmbedder) ChunkText(text string, _ int, _ int) ([]string, error) {
	return []string{text}, nil
}

// Dimensions returns the configured output dimensions.
func (e *VoyageEmbedder) Dimensions() int { return e.config.Dimensions }

// Model returns the configured provider model.
func (e *VoyageEmbedder) Model() string { return e.config.Model }

// Backend reports local HTTP marshalling, not the opaque remote GPU backend.
func (e *VoyageEmbedder) Backend() string { return "cpu" }

// ContentProperty identifies the explicitly structured multimodal property.
func (e *VoyageEmbedder) ContentProperty() string { return e.options.ContentProperty }

// EmbeddingSpace identifies the model/API/dimensions/endpoint used by this provider.
func (e *VoyageEmbedder) EmbeddingSpace() voyage.Space {
	return e.client.EmbeddingSpace(e.api, e.Model(), e.Dimensions())
}

// PrepareDocument preserves structured part order. A structured property is an
// authoritative content list: it is not flattened into BuildText or fetched here.
// Selecting contextualized mode for explicitly multimodal input fails visibly.
func (e *VoyageEmbedder) PrepareDocument(text string, properties map[string]any) (ManagedInput, error) {
	raw, structured := properties[e.ContentProperty()]
	if e.options.Mode == "contextualized" {
		if structured {
			return ManagedInput{}, fmt.Errorf("contextualized provider cannot consume structured multimodal content")
		}
		if strings.TrimSpace(text) == "" {
			return ManagedInput{}, fmt.Errorf("managed document text is empty")
		}
		return ManagedInput{Text: text}, nil
	}
	parts := []voyage.Part{{Type: "text", Text: text}}
	if structured {
		// Decode into fresh parts: reusing the seeded text element would retain
		// its Text field on an image object and create an invalid mixed part.
		parts = nil
		b, err := json.Marshal(raw)
		if encoded, ok := raw.(string); ok {
			b = []byte(encoded)
			err = nil
		}
		if err != nil {
			return ManagedInput{}, fmt.Errorf("invalid structured embedding content")
		}
		decoder := json.NewDecoder(bytes.NewReader(b))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&parts); err != nil {
			return ManagedInput{}, fmt.Errorf("invalid structured embedding content")
		}
	}
	req := voyage.MultimodalRequest{Inputs: []voyage.MultimodalInput{{Content: parts}}, Model: e.Model(), Purpose: voyage.Document, Dimensions: e.Dimensions(), Truncation: e.options.Truncation}
	if err := voyage.ValidateMultimodal(req); err != nil {
		return ManagedInput{}, err
	}
	return ManagedInput{Content: parts}, nil
}

// EmbedDocument generates a single whole-document result, with every contextual
// chunk retained. No storage writes, index updates, or independent worker run here.
func (e *VoyageEmbedder) EmbedDocument(ctx context.Context, input ManagedInput) (ManagedDocumentResult, error) {
	out := ManagedDocumentResult{Space: e.EmbeddingSpace(), InputFingerprint: input.Fingerprint()}
	if e.options.Mode == "contextualized" {
		if len(input.Content) > 0 {
			return out, fmt.Errorf("contextualized document must contain text only")
		}
		req := voyage.ContextRequest{Model: e.Model(), Purpose: voyage.Document, Dimensions: e.Dimensions(), AutoChunk: e.autoChunk}
		if e.autoChunk {
			req.ChunkSize = e.options.ChunkSize
			req.ChunkOverlap = e.options.ChunkOverlap
			req.Documents = []string{input.Text}
		} else {
			req.Chunks = [][]string{{input.Text}}
		}
		response, err := e.client.Contextualize(ctx, req)
		if err != nil {
			return out, err
		}
		out.Metadata = response.Metadata
		for _, chunk := range response.Documents[0].Chunks {
			out.Embeddings = append(out.Embeddings, chunk.Embedding)
			out.ChunkTexts = append(out.ChunkTexts, chunk.Text)
		}
		return out, nil
	}
	if input.Text != "" {
		return out, fmt.Errorf("multimodal document must contain structured parts")
	}
	response, err := e.client.Multimodal(ctx, voyage.MultimodalRequest{Inputs: []voyage.MultimodalInput{{Content: input.Content}}, Model: e.Model(), Purpose: voyage.Document, Dimensions: e.Dimensions(), Truncation: e.options.Truncation})
	if err != nil {
		return out, err
	}
	out.Metadata = response.Metadata
	out.Embeddings = response.Embeddings
	var text []string
	for _, part := range input.Content {
		if part.Type == "text" {
			text = append(text, part.Text)
		}
	}
	out.ChunkTexts = []string{strings.Join(text, "\n")}
	return out, nil
}

// EmbedQuery uses the matching model's query prompt and never provider auto-chunks
// or client-averages a query. Oversized queries fail at the provider boundary.
func (e *VoyageEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	if e.options.Mode == "contextualized" {
		response, err := e.client.Contextualize(ctx, voyage.ContextRequest{Documents: []string{text}, Model: e.Model(), Purpose: voyage.Query, Dimensions: e.Dimensions()})
		if err != nil {
			return nil, err
		}
		return response.Documents[0].Chunks[0].Embedding, nil
	}
	return e.EmbedMultimodalQuery(ctx, []voyage.Part{{Type: "text", Text: text}})
}

// EmbedMultimodalQuery supports image or interleaved queries in the SAME multimodal
// model space as its documents. It rejects contextualized providers explicitly.
func (e *VoyageEmbedder) EmbedMultimodalQuery(ctx context.Context, parts []voyage.Part) ([]float32, error) {
	if e.options.Mode != "multimodal" {
		return nil, fmt.Errorf("structured queries require the multimodal provider")
	}
	response, err := e.client.Multimodal(ctx, voyage.MultimodalRequest{Inputs: []voyage.MultimodalInput{{Content: parts}}, Model: e.Model(), Purpose: voyage.Query, Dimensions: e.Dimensions(), Truncation: e.options.Truncation})
	if err != nil {
		return nil, err
	}
	return response.Embeddings[0], nil
}
