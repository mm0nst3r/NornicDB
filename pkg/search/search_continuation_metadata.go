package search

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// searchPageMetadata copies additional explanatory/diagnostic JSON fields from
// ordinary retrieval without depending on optional retrieval implementations.
// Project before encoding: source properties, nodes, previews and vectors must
// neither enter retained cursors nor be serialized as temporary metadata.
func searchPageMetadata(value any, limit int64) (json.RawMessage, error) {
	v := reflect.ValueOf(value)
	if v.Kind() == reflect.Pointer {
		v = v.Elem()
	}
	fields := make(map[string]json.RawMessage)
	remaining := limit - 2 // surrounding object braces
	for i := 0; i < v.NumField(); i++ {
		field := v.Type().Field(i)
		tag := strings.Split(field.Tag.Get("json"), ",")
		name := tag[0]
		if !field.IsExported() || name == "" || name == "-" {
			continue
		}
		switch name {
		case "id", "nodeId", "type", "labels", "title", "description", "content_preview", "properties",
			"score", "similarity", "rrf_score", "vector_rank", "bm25_rank",
			"status", "query", "results", "total_candidates", "returned", "search_method", "fallback_triggered", "message", "metrics",
			"node", "nodes", "embedding", "embeddings", "vector", "vectors":
			continue
		}
		fv := v.Field(i)
		if len(tag) > 1 && tag[1] == "omitempty" && (fv.IsZero() || ((fv.Kind() == reflect.Slice || fv.Kind() == reflect.Map) && fv.Len() == 0)) {
			continue
		}
		encoded, err := json.Marshal(fv.Interface())
		if err != nil {
			return nil, fmt.Errorf("continuation metadata %s: %v: %w", name, err, ErrSearchPageRequest)
		}
		key, _ := json.Marshal(name)
		size := int64(len(key) + 1 + len(encoded))
		if len(fields) > 0 {
			size++ // comma
		}
		if size > remaining {
			return nil, ErrSearchContinuationLimit
		}
		remaining -= size
		fields[name] = encoded
	}
	if len(fields) == 0 {
		return nil, nil
	}
	return json.Marshal(fields)
}
