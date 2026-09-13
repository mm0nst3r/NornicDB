package storage

import (
	"encoding/json"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestManagedWALRecoveryPreservesControlAndConditionalResult(t *testing.T) {
	engine := NewMemoryEngine()
	defer engine.Close()
	original := &Node{ID: "document", Properties: map[string]any{"content": "source"}}
	apply := func(op OperationType, data WALNodeData) {
		raw, err := json.Marshal(data)
		require.NoError(t, err)
		require.NoError(t, ReplayWALEntry(engine, WALEntry{Operation: op, Database: "text", Data: raw}))
	}
	apply(OpCreateNode, WALNodeData{Node: original})
	running := CopyNode(original)
	running.EmbedMeta = map[string]any{"embedding_operation_id": "operation-1", "embedding_status": "running", "embedding_attempt_count": 1, "embedding_attempts": []map[string]any{{"operation_id": "operation-1", "outcome": "running"}}}
	apply(OpUpdateEmbeddingState, WALNodeData{Node: running, OldNode: original})
	complete := CopyNode(running)
	complete.ChunkEmbeddings = [][]float32{{1, 0}, {0, 1}}
	complete.EmbedMeta["chunk_texts"] = []string{"first", "second"}
	complete.EmbedMeta["embedding_status"] = "completed"
	apply(OpUpdateEmbeddingState, WALNodeData{Node: complete, OldNode: running})
	got, err := engine.GetNode("text:document")
	require.NoError(t, err)
	require.Equal(t, complete.ChunkEmbeddings, got.ChunkEmbeddings)
	require.Equal(t, "completed", got.EmbedMeta["embedding_status"])
	paused := CopyNode(complete)
	paused.EmbedMeta["embedding_control"] = "paused"
	paused.EmbedMeta["embedding_operation_id"] = "control-2"
	apply(OpUpdateNode, WALNodeData{Node: paused, OldNode: complete, TxID: "control-transaction"})
	apply(OpUpdateEmbeddingState, WALNodeData{Node: complete, OldNode: running})
	got, err = engine.GetNode("text:document")
	require.NoError(t, err)
	require.Equal(t, "paused", got.EmbedMeta["embedding_control"])
	require.Equal(t, "control-2", got.EmbedMeta["embedding_operation_id"])
	raw, err := json.Marshal(WALNodeData{Node: paused, OldNode: complete, TxID: "control-transaction"})
	require.NoError(t, err)
	var decoded WALNodeData
	require.NoError(t, json.Unmarshal(raw, &decoded))
	require.Equal(t, "control-transaction", decoded.TxID)
	require.NotNil(t, decoded.OldNode)
	public, err := json.Marshal(paused)
	require.NoError(t, err)
	require.NotContains(t, string(public), "embedding_control")
}
