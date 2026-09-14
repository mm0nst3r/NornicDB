package voyage

import (
	"context"
	"math"
	"strings"
)

// RerankRequest uses Voyage's top_k contract, not Cohere's top_n contract.
// TopK=0 returns all supplied documents. Truncation=false (the zero value)
// instructs Voyage to reject over-limit inputs rather than silently discard text.
// Exact model token limits are validated by the provider, not a byte heuristic.
type RerankRequest struct {
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	Model           string   `json:"model"`
	TopK            int      `json:"top_k,omitempty"`
	ReturnDocuments bool     `json:"return_documents"`
	Truncation      bool     `json:"truncation"`
}

// Ranking identifies an original input document. Order in RerankResponse.Rankings
// is exactly the provider's order, including ties and small score differences.
type Ranking struct {
	Index int     `json:"index"`
	Score float64 `json:"relevance_score"`
}

// RerankResponse contains verified results plus request-local diagnostics. Caller
// document identity/text is recovered from Ranking.Index, never a returned label.
type RerankResponse struct {
	Rankings []Ranking `json:"data"`
	Metadata Metadata  `json:"metadata"`
}

// Rerank reranks 1–1000 documents without slicing, re-sorting, or inventing missing
// scores. Incorrect, duplicate, missing, or out-of-range indices fail the call.
func (c *Client) Rerank(ctx context.Context, req RerankRequest) (*RerankResponse, error) {
	if strings.TrimSpace(req.Model) == "" || strings.TrimSpace(req.Query) == "" ||
		len(req.Documents) == 0 || len(req.Documents) > 1000 || req.TopK < 0 || req.TopK > len(req.Documents) {
		return nil, failure(InvalidInput, Metadata{})
	}
	for _, text := range req.Documents {
		if strings.TrimSpace(text) == "" {
			return nil, failure(InvalidInput, Metadata{})
		}
	}
	var wire struct {
		envelope
		Data []struct {
			Index *int     `json:"index"`
			Score *float64 `json:"relevance_score"`
		} `json:"data"`
	}
	meta, err := c.post(ctx, "/rerank", req, &wire)
	if err != nil {
		return nil, err
	}
	meta, err = wire.envelope.metadata(meta, req.Model)
	if err != nil {
		return nil, err
	}
	want := req.TopK
	if want == 0 {
		want = len(req.Documents)
	}
	if len(wire.Data) != want {
		return nil, failure(MalformedResponse, meta)
	}
	out := &RerankResponse{Metadata: meta, Rankings: make([]Ranking, 0, want)}
	seen := make([]bool, len(req.Documents))
	for _, row := range wire.Data {
		if row.Index == nil || row.Score == nil || *row.Index < 0 || *row.Index >= len(seen) ||
			seen[*row.Index] || math.IsNaN(*row.Score) || math.IsInf(*row.Score, 0) {
			return nil, failure(MalformedResponse, meta)
		}
		seen[*row.Index] = true
		out.Rankings = append(out.Rankings, Ranking{Index: *row.Index, Score: *row.Score})
	}
	return out, nil
}
