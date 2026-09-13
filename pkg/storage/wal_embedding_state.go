package storage

import "encoding/json"

// Node's public JSON intentionally hides generated state. WAL records must retain
// it for accepted results, controls and attempts to survive recovery.
type walNodeAlias Node
type walNodeState struct {
	walNodeAlias
	EmbedMeta       map[string]any       `json:"embedding_metadata,omitempty"`
	ChunkEmbeddings [][]float32          `json:"chunk_embeddings,omitempty"`
	NamedEmbeddings map[string][]float32 `json:"named_embeddings,omitempty"`
}

func nodeWALState(node *Node) *walNodeState {
	if node == nil {
		return nil
	}
	return &walNodeState{walNodeAlias: walNodeAlias(*node), EmbedMeta: node.EmbedMeta, ChunkEmbeddings: node.ChunkEmbeddings, NamedEmbeddings: node.NamedEmbeddings}
}
func (state *walNodeState) node() *Node {
	if state == nil {
		return nil
	}
	node := (*Node)(&state.walNodeAlias)
	node.EmbedMeta = state.EmbedMeta
	node.ChunkEmbeddings = state.ChunkEmbeddings
	node.NamedEmbeddings = state.NamedEmbeddings
	return node
}

// MarshalJSON retains native state in the WAL's private node representation.
func (data WALNodeData) MarshalJSON() ([]byte, error) {
	return json.Marshal(walNodeRecord{Node: nodeWALState(data.Node), OldNode: nodeWALState(data.OldNode), TxID: data.TxID})
}

type walNodeRecord struct {
	Node    *walNodeState `json:"node"`
	OldNode *walNodeState `json:"old_node,omitempty"`
	TxID    string        `json:"tx_id,omitempty"`
}

// UnmarshalJSON decodes the WAL node state without changing public Node JSON.
func (data *WALNodeData) UnmarshalJSON(raw []byte) error {
	var decoded walNodeRecord
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	data.Node = decoded.Node.node()
	data.OldNode = decoded.OldNode.node()
	data.TxID = decoded.TxID
	return nil
}
