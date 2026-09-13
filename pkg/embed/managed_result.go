package embed

import (
	"fmt"
	"github.com/orneryd/nornicdb/pkg/embeddingutil"
	"github.com/orneryd/nornicdb/pkg/storage"
	"time"
)

// PrepareManagedNode preserves structured content and uses the same selected
// textual properties for background and explicit WITH EMBEDDING generation.
func PrepareManagedNode(provider ManagedDocumentEmbedder, node *storage.Node, options *embeddingutil.EmbedTextOptions) (ManagedInput, error) {
	opts := embeddingutil.EmbedTextOptions{IncludeLabels: true}
	if options != nil {
		opts = *options
	}
	opts.Exclude = append(append([]string(nil), opts.Exclude...), provider.ContentProperty())
	return provider.PrepareDocument(embeddingutil.BuildText(node.Properties, node.Labels, &opts), node.Properties)
}

// ApplyManagedDocumentResult owns storage metadata for the complete provider
// result and completes its already-started job through the lifecycle owner.
// Callers persist the combined result atomically with the source-bound state.
func ApplyManagedDocumentResult(node *storage.Node, result ManagedDocumentResult) error {
	if len(result.Embeddings) == 0 || len(result.ChunkTexts) != len(result.Embeddings) {
		return fmt.Errorf("managed embedding result has inconsistent chunk metadata")
	}
	embeddingutil.ApplyManagedEmbedding(node, result.Embeddings, result.Space.Model, result.Space.Dimensions, time.Now())
	if node.EmbedMeta == nil {
		node.EmbedMeta = make(map[string]any)
	}
	node.EmbedMeta["embedding_provider"] = "voyage"
	node.EmbedMeta["embedding_space"] = result.Space.Key()
	node.EmbedMeta["embedding_api"] = result.Space.API
	node.EmbedMeta["embedding_input_fingerprint"] = result.InputFingerprint
	node.EmbedMeta["chunk_texts"] = append([]string(nil), result.ChunkTexts...)
	node.EmbedMeta["chunker_version"] = result.Metadata.ChunkerVersion
	delete(node.EmbedMeta, "embedding_usage")
	if result.Metadata.Usage != nil {
		node.EmbedMeta["embedding_usage"] = map[string]any{"total_tokens": result.Metadata.Usage.TotalTokens, "text_tokens": result.Metadata.Usage.TextTokens, "image_pixels": result.Metadata.Usage.ImagePixels}
	}
	usage, _ := node.EmbedMeta["embedding_usage"].(map[string]any)
	return embeddingutil.ApplyEmbeddingWorkEvent(node, embeddingutil.EmbeddingWorkEvent{
		Action: "complete", RequestID: result.Metadata.RequestID,
		HTTPStatus: result.Metadata.StatusCode, Usage: usage,
	})
}
