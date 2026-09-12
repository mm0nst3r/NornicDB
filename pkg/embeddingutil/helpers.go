package embeddingutil

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/orneryd/nornicdb/pkg/config"
	"github.com/orneryd/nornicdb/pkg/storage"
)

// EmbedTextOptions controls which properties and labels are used when building embedding text.
type EmbedTextOptions struct {
	Include       []string
	Exclude       []string
	IncludeLabels bool
}

// BuildText creates canonical embedding text from node properties and labels.
func BuildText(properties map[string]interface{}, labels []string, opts *EmbedTextOptions) string {
	if opts == nil {
		opts = &EmbedTextOptions{IncludeLabels: true}
	}
	var parts []string

	if opts.IncludeLabels && len(labels) > 0 {
		parts = append(parts, fmt.Sprintf("labels: %s", strings.Join(labels, ", ")))
	}

	excludeSet := make(map[string]bool, len(metadataPropertyKeys)+len(opts.Exclude))
	for key := range metadataPropertyKeys {
		excludeSet[key] = true
	}
	for _, key := range opts.Exclude {
		excludeSet[key] = true
	}

	var includeSet map[string]bool
	if len(opts.Include) > 0 {
		includeSet = make(map[string]bool, len(opts.Include))
		for _, key := range opts.Include {
			includeSet[key] = true
		}
	}

	// Stable property ordering is required for source fingerprints and retries.
	keys := make([]string, 0, len(properties))
	for key := range properties {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		val := properties[key]
		if excludeSet[key] {
			continue
		}
		if includeSet != nil && !includeSet[key] {
			continue
		}

		var strVal string
		switch v := val.(type) {
		case string:
			strVal = v
		case []interface{}:
			strs := make([]string, 0, len(v))
			for _, item := range v {
				if s, ok := item.(string); ok {
					strs = append(strs, s)
				} else {
					strs = append(strs, fmt.Sprintf("%v", item))
				}
			}
			strVal = strings.Join(strs, ", ")
		case bool:
			strVal = fmt.Sprintf("%v", v)
		case int, int64, float64:
			strVal = fmt.Sprintf("%v", v)
		case nil:
			strVal = "null"
		default:
			if b, err := json.Marshal(v); err == nil {
				strVal = string(b)
			} else {
				strVal = fmt.Sprintf("%v", v)
			}
		}
		parts = append(parts, fmt.Sprintf("%s: %s", key, strVal))
	}

	result := strings.Join(parts, "\n")
	if result == "" {
		return "node"
	}
	return result
}

// IsMetadataPropertyKey reports whether a property key is internal embedding metadata.
func IsMetadataPropertyKey(key string) bool {
	return metadataPropertyKeys[key]
}

// InvalidateManagedEmbeddings clears worker-managed embedding state on a node.
func InvalidateManagedEmbeddings(node *storage.Node) {
	if node == nil {
		return
	}
	_ = ApplyEmbeddingWorkEvent(node, EmbeddingWorkEvent{Action: "invalidate"})
}

// EmbedTextOptionsFromConfig maps runtime embedding worker config to text builder options.
func EmbedTextOptionsFromConfig(cfg *config.Config) *EmbedTextOptions {
	if cfg == nil {
		return &EmbedTextOptions{IncludeLabels: true}
	}
	return EmbedTextOptionsFromFields(
		cfg.EmbeddingWorker.PropertiesInclude,
		cfg.EmbeddingWorker.PropertiesExclude,
		cfg.EmbeddingWorker.IncludeLabels,
	)
}

// EmbedTextOptionsFromFields builds text options from raw include/exclude settings.
func EmbedTextOptionsFromFields(include []string, exclude []string, includeLabels bool) *EmbedTextOptions {
	return &EmbedTextOptions{
		Include:       include,
		Exclude:       exclude,
		IncludeLabels: includeLabels,
	}
}

// ApplyManagedEmbedding writes worker-compatible embedding payload to node fields.
func ApplyManagedEmbedding(node *storage.Node, embeddings [][]float32, model string, dimensions int, embeddedAt time.Time) {
	if node == nil {
		return
	}
	node.ChunkEmbeddings = embeddings
	if node.EmbedMeta == nil {
		node.EmbedMeta = make(map[string]any)
	}
	node.EmbedMeta["chunk_count"] = len(embeddings)
	node.EmbedMeta["embedding_model"] = model
	node.EmbedMeta["embedding_dimensions"] = dimensions
	node.EmbedMeta["has_embedding"] = len(embeddings) > 0
	node.EmbedMeta["embedded_at"] = embeddedAt.Format(time.RFC3339)
}

var metadataPropertyKeys = map[string]bool{
	"embedding":            true,
	"has_embedding":        true,
	"embedding_skipped":    true,
	"embedding_model":      true,
	"embedding_dimensions": true,
	"embedded_at":          true,
	"has_chunks":           true,
	"chunk_count":          true,
	"createdAt":            true,
	"updatedAt":            true,
	"id":                   true,
}
