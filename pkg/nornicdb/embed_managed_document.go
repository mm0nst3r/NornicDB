package nornicdb

import (
	"errors"
	"fmt"
	"time"

	"github.com/orneryd/nornicdb/pkg/embed"
	"github.com/orneryd/nornicdb/pkg/embeddingutil"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/orneryd/nornicdb/pkg/voyage"
)

// managedNodeInput excludes structured content from legacy text flattening.
// The provider consumes that property separately, preserving image representation
// and order. This function performs no network or storage side effects.
func (ew *EmbedWorker) managedNodeInput(provider embed.ManagedDocumentEmbedder, node *storage.Node) (embed.ManagedInput, error) {
	opts := embeddingutil.EmbedTextOptionsFromFields(ew.config.PropertiesInclude, ew.config.PropertiesExclude, ew.config.IncludeLabels)
	return embed.PrepareManagedNode(provider, node, opts)
}

func (ew *EmbedWorker) persistEmbeddingState(node, expected *storage.Node) error {
	updater, ok := ew.storage.(storage.ConditionalEmbeddingUpdater)
	if !ok {
		return fmt.Errorf("storage does not support conditional embedding publication")
	}
	return updater.UpdateNodeEmbeddingIfCurrent(node, expected)
}

func (ew *EmbedWorker) beginManagedAttempt(node *storage.Node, model string) error {
	expected := storage.CopyNode(node)
	if embeddingutil.EmbeddingAttemptCount(node) >= ew.config.MaxRetries {
		if err := embeddingutil.ApplyEmbeddingWorkEvent(node, embeddingutil.EmbeddingWorkEvent{Action: "fail", ErrorCode: "attempt_limit"}); err != nil {
			return err
		}
		if err := ew.persistEmbeddingState(node, expected); err != nil {
			return err
		}
		return fmt.Errorf("managed embedding attempt limit reached")
	}
	if err := embeddingutil.ApplyEmbeddingWorkEvent(node, embeddingutil.EmbeddingWorkEvent{Action: "begin", Provider: "voyage", Model: model}); err != nil {
		return err
	}
	if err := ew.persistEmbeddingState(node, expected); err != nil {
		return err
	}
	ew.markNodeEmbedded(node.ID)
	return nil
}

func managedUsage(metadata voyage.Metadata) map[string]any {
	if metadata.Usage == nil {
		return nil
	}
	return map[string]any{"total_tokens": metadata.Usage.TotalTokens, "text_tokens": metadata.Usage.TextTokens, "image_pixels": metadata.Usage.ImagePixels}
}

func (ew *EmbedWorker) recordManagedFailure(node, expected *storage.Node, cause error) error {
	event := embeddingutil.EmbeddingWorkEvent{Action: "fail", ErrorCode: "invalid_input"}
	var provider *voyage.Error
	if errors.As(cause, &provider) {
		event.ErrorCode = provider.Kind
		event.RequestID = provider.Metadata.RequestID
		event.HTTPStatus = provider.Metadata.StatusCode
		event.Usage = managedUsage(provider.Metadata)
		if provider.Retryable() && embeddingutil.EmbeddingAttemptCount(node) < ew.config.MaxRetries {
			delay := provider.Metadata.RetryAfter
			backoff := time.Duration(embeddingutil.EmbeddingAttemptCount(node)) * 2 * time.Second
			if delay < backoff {
				delay = backoff
			}
			event.RetryAt = time.Now().Add(delay)
		}
	}
	if ew.ctx.Err() != nil && embeddingutil.EmbeddingAttemptCount(node) < ew.config.MaxRetries {
		event.ErrorCode = "interrupted"
		event.RetryAt = time.Now()
	}
	if err := embeddingutil.ApplyEmbeddingWorkEvent(node, event); err != nil {
		return err
	}
	return ew.persistEmbeddingState(node, expected)
}
