package search

import "encoding/json"

// ResultMetadata exposes provider evidence without coupling result consumers to a provider.
func (r SearchResult) ResultMetadata() map[string]any {
	if len(r.SupportingPassages) == 0 {
		return nil
	}
	return map[string]any{"supporting_passages": PassageMaps(r.SupportingPassages)}
}

// ResultMetadata exposes request-local provider outcomes to response consumers.
func (r *SearchResponse) ResultMetadata() map[string]any {
	if r == nil || r.Rerank == nil {
		return nil
	}
	return map[string]any{"rerank": r.Rerank.Map()}
}

func (r *SearchResponse) ResponseHeaders() map[string]string {
	if r == nil || r.Rerank == nil {
		return nil
	}
	value, _ := json.Marshal(r.Rerank)
	return map[string]string{"X-NornicDB-Rerank": string(value)}
}

func (r *SearchResponse) ResponseTrailers() map[string]string {
	if r == nil || r.Rerank == nil {
		return nil
	}
	value, _ := json.Marshal(r.Rerank)
	return map[string]string{"nornicdb-rerank": string(value)}
}
