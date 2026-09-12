package replication

import (
	"fmt"
	"github.com/orneryd/nornicdb/pkg/storage"
	"time"
)

type conditionalEmbeddingPayload struct{ Node, Expected []byte }

// FindNodeNeedingEmbedding preserves the local pending-index read capability.
// The worker checks leadership before claiming work; accepted state writes use
// UpdateNodeEmbeddingIfCurrent and are replicated.
func (e *ReplicatedEngine) FindNodeNeedingEmbedding() *storage.Node {
	if finder, ok := e.Engine.(interface{ FindNodeNeedingEmbedding() *storage.Node }); ok {
		return finder.FindNodeNeedingEmbedding()
	}
	return storage.FindNodeNeedingEmbedding(e.Engine)
}

// RefreshPendingEmbeddingsIndex refreshes the local derived work index.
func (e *ReplicatedEngine) RefreshPendingEmbeddingsIndex() int {
	if index, ok := e.Engine.(interface{ RefreshPendingEmbeddingsIndex() int }); ok {
		return index.RefreshPendingEmbeddingsIndex()
	}
	return 0
}

// MarkNodeEmbedded removes a local pending-index entry after claim/completion.
func (e *ReplicatedEngine) MarkNodeEmbedded(id storage.NodeID) {
	if index, ok := e.Engine.(interface{ MarkNodeEmbedded(storage.NodeID) }); ok {
		index.MarkNodeEmbedded(id)
	}
}

// AddToPendingEmbeddings returns work to the local derived pending index.
func (e *ReplicatedEngine) AddToPendingEmbeddings(id storage.NodeID) {
	if index, ok := e.Engine.(interface{ AddToPendingEmbeddings(storage.NodeID) }); ok {
		index.AddToPendingEmbeddings(id)
	}
}

// UpdateNodeEmbeddingIfCurrent replicates one conditional publication. Followers
// apply the same comparison and never run the provider to reconstruct the result.
func (e *ReplicatedEngine) UpdateNodeEmbeddingIfCurrent(node, expected *storage.Node) error {
	payload := conditionalEmbeddingPayload{}
	var err error
	payload.Node, err = encodeNodePayload(node)
	if err != nil {
		return err
	}
	payload.Expected, err = encodeNodePayload(expected)
	if err != nil {
		return err
	}
	data, err := encodeGob(payload)
	if err != nil {
		return err
	}
	return e.replicator.Apply(&Command{Type: CmdUpdateEmbeddingIfCurrent, Data: data, Timestamp: time.Now()}, e.timeout)
}

func (a *StorageAdapter) applyConditionalEmbedding(data []byte) error {
	var payload conditionalEmbeddingPayload
	if err := decodeGob(data, &payload); err != nil {
		return err
	}
	node, err := decodeNodePayload(payload.Node)
	if err != nil {
		return err
	}
	expected, err := decodeNodePayload(payload.Expected)
	if err != nil {
		return err
	}
	updater, ok := a.engine.(storage.ConditionalEmbeddingUpdater)
	if !ok {
		return fmt.Errorf("replicated storage does not support conditional embedding publication")
	}
	return updater.UpdateNodeEmbeddingIfCurrent(node, expected)
}
