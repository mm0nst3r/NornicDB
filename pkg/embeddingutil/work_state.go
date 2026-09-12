package embeddingutil

import (
	"fmt"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/orneryd/nornicdb/pkg/storage"
)

// EmbeddingWorkEvent describes a transition of one source-bound embedding job.
// Provider diagnostics must already be sanitized; no submitted content belongs here.
type EmbeddingWorkEvent struct {
	Action     string
	Provider   string
	Model      string
	RequestID  string
	ErrorCode  string
	HTTPStatus int
	RetryAt    time.Time
	Usage      map[string]any
}

// EmbeddingAttemptCount returns the current job's attempt count across restarts.
func EmbeddingAttemptCount(node *storage.Node) int {
	if !storage.EmbeddingAttemptSourceCurrent(node) {
		return 0
	}
	value, _ := strconv.Atoi(fmt.Sprint(node.EmbedMeta["embedding_attempt_count"]))
	return value
}

// ApplyEmbeddingWorkEvent owns the managed job's status, control and attempt
// records. The caller must persist the result through its transaction or a
// conditional embedding update before treating the transition as accepted.
func ApplyEmbeddingWorkEvent(node *storage.Node, event EmbeddingWorkEvent) error {
	if node == nil {
		return storage.ErrInvalidData
	}
	if event.Action == "invalidate" {
		control, _ := node.EmbedMeta["embedding_control"].(string)
		operation, _ := node.EmbedMeta["embedding_operation_id"].(string)
		node.ChunkEmbeddings = nil
		node.EmbedMeta = nil
		if control != "" {
			node.EmbedMeta = map[string]any{"embedding_control": control, "embedding_status": control, "embedding_operation_id": operation}
		}
		return nil
	}
	if node.EmbedMeta == nil {
		node.EmbedMeta = make(map[string]any)
	}
	meta := node.EmbedMeta
	status, _ := meta["embedding_status"].(string)
	control, _ := meta["embedding_control"].(string)
	attempts, err := embeddingAttempts(meta["embedding_attempts"])
	if err != nil {
		return err
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	switch event.Action {
	case "begin":
		if control == "paused" || control == "cancelled" {
			return fmt.Errorf("embedding work is %s", control)
		}
		if status == "running" && len(attempts) > 0 {
			attempts[len(attempts)-1]["outcome"] = "interrupted"
			attempts[len(attempts)-1]["error_code"] = "process_interrupted"
			attempts[len(attempts)-1]["finished_at"] = now
		}
		operation := uuid.NewString()
		meta["embedding_operation_id"] = operation
		meta["embedding_attempt_count"] = EmbeddingAttemptCount(node) + 1
		fingerprint, err := storage.EmbeddingSourceFingerprint(node)
		if err != nil {
			return err
		}
		meta["embedding_attempt_source_fingerprint"] = fingerprint
		attempts = append(attempts, map[string]any{"operation_id": operation, "source_fingerprint": fingerprint, "provider": event.Provider, "model": event.Model, "started_at": now, "outcome": "running"})
		status = "running"
		delete(meta, "embedding_retry_at")
	case "complete", "fail":
		if len(attempts) == 0 {
			return fmt.Errorf("embedding attempt was not started")
		}
		attempt := attempts[len(attempts)-1]
		attempt["finished_at"] = now
		attempt["request_id"] = event.RequestID
		attempt["http_status"] = event.HTTPStatus
		if event.Usage != nil {
			attempt["usage"] = event.Usage
		}
		if event.Action == "complete" {
			status = "completed"
			attempt["outcome"] = "completed"
			delete(meta, "embedding_error_code")
			delete(meta, "embedding_retry_at")
			fingerprint, err := storage.EmbeddingSourceFingerprint(node)
			if err != nil {
				return err
			}
			meta["embedding_source_fingerprint"] = fingerprint
		} else {
			status = "failed"
			attempt["outcome"] = "failed"
			attempt["error_code"] = event.ErrorCode
			meta["embedding_error_code"] = event.ErrorCode
			if !event.RetryAt.IsZero() {
				status = "pending"
				attempt["outcome"] = "retry_scheduled"
				meta["embedding_retry_at"] = event.RetryAt.UTC().Format(time.RFC3339Nano)
			} else {
				delete(meta, "embedding_retry_at")
			}
		}
		meta["embedding_request_id"] = event.RequestID
		meta["embedding_http_status"] = event.HTTPStatus
	case "pause", "cancel":
		selected := "paused"
		if event.Action == "cancel" {
			selected = "cancelled"
		}
		if control == selected {
			return nil
		}
		control = selected
		meta["embedding_operation_id"] = uuid.NewString()
		if !storage.ManagedEmbeddingCurrent(node) {
			status = selected
		}
		if len(attempts) > 0 && attempts[len(attempts)-1]["outcome"] == "running" {
			attempts[len(attempts)-1]["outcome"] = selected
			attempts[len(attempts)-1]["finished_at"] = now
		}
	case "resume":
		if control == "" {
			return nil
		}
		control = ""
		meta["embedding_operation_id"] = uuid.NewString()
		if storage.ManagedEmbeddingCurrent(node) {
			status = "completed"
		} else {
			status = "pending"
		}
	case "retry":
		if status != "failed" {
			return fmt.Errorf("only failed embedding work can be retried")
		}
		control = ""
		status = "pending"
		meta["embedding_operation_id"] = uuid.NewString()
		meta["embedding_attempt_count"] = 0
		delete(meta, "embedding_retry_at")
		delete(meta, "embedding_error_code")
	default:
		return fmt.Errorf("unknown embedding action")
	}
	meta["embedding_status"] = status
	meta["embedding_control"] = control
	meta["embedding_attempts"] = attempts
	meta["embedding_state_updated_at"] = now
	return nil
}

func embeddingAttempts(value any) ([]map[string]any, error) {
	var input []map[string]any
	switch rows := value.(type) {
	case nil:
		return nil, nil
	case []map[string]any:
		input = rows
	case []any:
		input = make([]map[string]any, 0, len(rows))
		for _, row := range rows {
			attempt, ok := row.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("invalid embedding attempt metadata")
			}
			input = append(input, attempt)
		}
	default:
		return nil, fmt.Errorf("invalid embedding attempt metadata")
	}
	result := make([]map[string]any, len(input))
	for i, row := range input {
		result[i] = make(map[string]any, len(row))
		for key, value := range row {
			if nested, ok := value.(map[string]any); ok {
				copy := make(map[string]any, len(nested))
				for k, v := range nested {
					copy[k] = v
				}
				value = copy
			}
			result[i][key] = value
		}
	}
	return result, nil
}

// EmbeddingWorkStatus returns client-safe job diagnostics, including a derived
// pending state after a source edit. Chunk text and vectors are retrieval data.
func EmbeddingWorkStatus(node *storage.Node) map[string]any {
	result := map[string]any{"node_id": string(node.ID), "status": "pending", "control": "", "current_result": storage.ManagedEmbeddingCurrent(node)}
	if count, ok := node.EmbedMeta["chunk_count"]; ok {
		result["chunk_count"] = count
	}
	for _, name := range []string{"status", "control", "model", "dimensions", "operation_id", "attempt_count", "retry_at", "error_code", "request_id", "http_status", "attempts", "space"} {
		if value, ok := node.EmbedMeta["embedding_"+name]; ok {
			result[name] = value
		}
	}
	if result["current_result"] == true {
		result["status"] = "completed"
	} else if result["status"] == "completed" || !storage.EmbeddingAttemptSourceCurrent(node) {
		result["status"] = "pending"
	}
	if control, _ := result["control"].(string); control != "" && result["current_result"] != true {
		result["status"] = control
	}
	if attempts, err := embeddingAttempts(node.EmbedMeta["embedding_attempts"]); err == nil {
		result["attempts"] = attempts
	}
	return result
}
