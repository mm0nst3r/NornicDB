package cypher

import (
	"context"
	"fmt"

	"github.com/orneryd/nornicdb/pkg/embeddingutil"
	"github.com/orneryd/nornicdb/pkg/storage"
)

func (e *StorageExecutor) callDbEmbeddingWork(ctx context.Context, query string, control bool) (*ExecuteResult, error) {
	name := "DB.EMBEDDING.STATUS"
	if control {
		name = "DB.EMBEDDING.CONTROL"
	}
	request, err := e.parseRagProcedureRequest(ctx, query, name)
	if err != nil {
		return nil, err
	}
	id, ok := request["nodeId"].(string)
	if !ok || id == "" {
		return nil, fmt.Errorf("nodeId is required")
	}
	store := e.getStorage(ctx)
	node, err := store.GetNode(storage.NodeID(normalizeNodeIDValue(id).(string)))
	if err != nil {
		return nil, err
	}
	if control {
		action, ok := request["action"].(string)
		if !ok {
			return nil, fmt.Errorf("embedding action is required")
		}
		switch action {
		case "pause", "resume", "cancel", "retry":
		default:
			return nil, fmt.Errorf("embedding action must be pause, resume, cancel or retry")
		}
		node = storage.CopyNode(node)
		if err := embeddingutil.ApplyEmbeddingWorkEvent(node, embeddingutil.EmbeddingWorkEvent{Action: action}); err != nil {
			return nil, err
		}
		if err := store.UpdateNode(node); err != nil {
			return nil, err
		}
	}
	return &ExecuteResult{Columns: []string{"status"}, Rows: [][]interface{}{{embeddingutil.EmbeddingWorkStatus(node)}}}, nil
}
