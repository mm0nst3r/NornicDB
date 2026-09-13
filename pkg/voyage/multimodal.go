package voyage

import (
	"context"
	"encoding/base64"
	"net/url"
	"strings"
)

// Part is exactly one text, image URL, or base64 image data URL. The client sends
// image representations structurally; it never embeds an image URL as plain text
// or fetches that URL locally. Video is intentionally not in this initial surface.
// Base64 image data URLs support PNG, JPEG, WEBP and GIF; decoded images and URL
// resources are still validated by Voyage for pixels, MIME, and model token limits.
type Part struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	ImageURL    string `json:"image_url,omitempty"`
	ImageBase64 string `json:"image_base64,omitempty"`
}

// MultimodalInput preserves the interleaved order of text and image parts.
type MultimodalInput struct {
	Content []Part `json:"content"`
}

// MultimodalRequest accepts text-only queries or structured documents/queries.
// Truncation is always serialized, with the safe zero value false.
type MultimodalRequest struct {
	Inputs     []MultimodalInput `json:"inputs"`
	Model      string            `json:"model"`
	Purpose    Purpose           `json:"input_type"`
	Dimensions int               `json:"output_dimension"`
	Truncation bool              `json:"truncation"`
}

// MultimodalResponse returns vectors in original input order and their model space.
type MultimodalResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Space      Space       `json:"space"`
	Metadata   Metadata    `json:"metadata"`
}

// Multimodal sends up to 1000 structured inputs, rejecting malformed one-of parts,
// mixed URL/base64 image representations, and invalid/incomplete output vectors.
func (c *Client) Multimodal(ctx context.Context, req MultimodalRequest) (*MultimodalResponse, error) {
	if err := ValidateMultimodal(req); err != nil {
		return nil, err
	}
	var wire struct {
		envelope
		Data []struct {
			Index     *int       `json:"index"`
			Embedding vectorWire `json:"embedding"`
		} `json:"data"`
	}
	meta, err := c.post(ctx, "/multimodalembeddings", req, &wire)
	if err != nil {
		return nil, err
	}
	meta, err = wire.envelope.metadata(meta, req.Model)
	if err != nil {
		return nil, err
	}
	if len(wire.Data) != len(req.Inputs) {
		return nil, failure(MalformedResponse, meta)
	}
	out := &MultimodalResponse{Embeddings: make([][]float32, len(req.Inputs)),
		Space: c.EmbeddingSpace(MultimodalAPI, req.Model, req.Dimensions), Metadata: meta}
	for _, row := range wire.Data {
		if row.Index == nil || *row.Index < 0 || *row.Index >= len(out.Embeddings) ||
			out.Embeddings[*row.Index] != nil || !validVector(row.Embedding, req.Dimensions) {
			return nil, failure(MalformedResponse, meta)
		}
		out.Embeddings[*row.Index] = []float32(row.Embedding)
	}
	return out, nil
}

// ValidateMultimodal validates structure and deterministic size limits without
// making a request. Token counts/pixel counts are provider-authoritative, not
// guessed from text length. ImageBase64's size cap is 20,000,000 decoded bytes.
func ValidateMultimodal(req MultimodalRequest) error {
	invalid := func() error { return failure(InvalidInput, Metadata{}) }
	if !validModelDimensions(req.Model, req.Dimensions) || len(req.Inputs) == 0 || len(req.Inputs) > 1000 ||
		(req.Purpose != Query && req.Purpose != Document) {
		return invalid()
	}
	usesURL, usesBase64 := false, false
	for _, input := range req.Inputs {
		if len(input.Content) == 0 {
			return invalid()
		}
		for _, part := range input.Content {
			switch part.Type {
			case "text":
				if strings.TrimSpace(part.Text) == "" || part.ImageURL != "" || part.ImageBase64 != "" {
					return invalid()
				}
			case "image_url":
				if part.Text != "" || part.ImageBase64 != "" {
					return invalid()
				}
				u, err := url.Parse(part.ImageURL)
				if err != nil || u.Hostname() == "" || u.User != nil || u.Fragment != "" ||
					(u.Scheme != "https" && u.Scheme != "http") {
					return invalid()
				}
				usesURL = true
			case "image_base64":
				if part.Text != "" || part.ImageURL != "" || !validImageDataURL(part.ImageBase64) {
					return invalid()
				}
				usesBase64 = true
			default:
				return invalid()
			}
		}
	}
	if usesURL && usesBase64 {
		return invalid()
	}
	return nil
}

func validImageDataURL(s string) bool {
	header, data, ok := strings.Cut(s, ",")
	if !ok || data == "" {
		return false
	}
	switch header {
	case "data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64", "data:image/gif;base64":
	default:
		return false
	}
	const maxImageBytes = 20_000_000
	if len(data) > base64.StdEncoding.EncodedLen(maxImageBytes) {
		return false
	}
	decoded, err := base64.StdEncoding.Strict().DecodeString(data)
	return err == nil && len(decoded) > 0 && len(decoded) <= maxImageBytes
}
