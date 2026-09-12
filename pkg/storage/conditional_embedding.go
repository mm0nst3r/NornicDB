package storage

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/dgraph-io/badger/v4"
	"github.com/orneryd/nornicdb/pkg/config"
)

// ErrEmbeddingSourceChanged means a generated result no longer belongs to the
// current source or operation. It must be discarded, never attached to that node.
var ErrEmbeddingSourceChanged = errors.New("embedding source or operation changed")

// ConditionalEmbeddingUpdater persists only embedding fields when the source and
// operation still match expected. It never creates a missing node.
type ConditionalEmbeddingUpdater interface {
	UpdateNodeEmbeddingIfCurrent(node, expected *Node) error
}

// EmbeddingSourceFingerprint identifies source inputs without storing their text
// in diagnostics. Named vectors participate so explicit user vectors supersede
// an in-flight managed generation. Embedding metadata is checked separately.
func EmbeddingSourceFingerprint(node *Node) (string, error) {
	if node == nil {
		return "", ErrInvalidData
	}
	labels := node.Labels
	if labels == nil {
		labels = []string{}
	}
	properties := node.Properties
	if properties == nil {
		properties = map[string]any{}
	}
	named := node.NamedEmbeddings
	if len(named) == 0 {
		named = nil
	}
	body, err := json.Marshal(struct {
		Labels     []string
		Properties map[string]any
		Named      map[string][]float32
	}{labels, properties, named})
	if err != nil {
		return "", fmt.Errorf("cannot fingerprint embedding source")
	}
	return fmt.Sprintf("%x", sha256.Sum256(body)), nil
}

// ManagedEmbeddingCurrent reports whether stored vectors belong to the current
// source. Existing embeddings without a recorded source fingerprint retain their
// established contract; newly managed results always record the fingerprint.
func ManagedEmbeddingCurrent(node *Node) bool {
	if node == nil || len(node.ChunkEmbeddings) == 0 || len(node.ChunkEmbeddings[0]) == 0 {
		return false
	}
	recorded, _ := node.EmbedMeta["embedding_source_fingerprint"].(string)
	if recorded == "" {
		return true
	}
	actual, err := EmbeddingSourceFingerprint(node)
	return err == nil && actual == recorded
}

func sameEmbeddingSource(current, expected *Node) (bool, error) {
	if current == nil || expected == nil || current.ID != expected.ID {
		return false, nil
	}
	currentHash, err := EmbeddingSourceFingerprint(current)
	if err != nil {
		return false, err
	}
	expectedHash, err := EmbeddingSourceFingerprint(expected)
	if err != nil {
		return false, err
	}
	if currentHash != expectedHash {
		return false, nil
	}
	for _, key := range []string{"embedding_operation_id", "embedding_control"} {
		left, _ := current.EmbedMeta[key].(string)
		right, _ := expected.EmbedMeta[key].(string)
		if left != right {
			return false, nil
		}
	}
	return true, nil
}

// Notifications are deferred through local wrappers so no callback runs while
// AsyncEngine holds its cache/flush guards. Only the outer public call emits them.
type deferredEmbeddingUpdater interface {
	updateEmbeddingDeferred(node, expected *Node) (func(), error)
}

func updateEmbeddingDeferred(engine Engine, node, expected *Node) (func(), error) {
	updater, ok := engine.(deferredEmbeddingUpdater)
	if !ok {
		return nil, ErrNotImplemented
	}
	return updater.updateEmbeddingDeferred(node, expected)
}

func finishEmbeddingUpdate(notify func(), err error) error {
	if err == nil && notify != nil {
		notify()
	}
	return err
}

// UpdateNodeEmbeddingIfCurrent atomically accepts a generated result bound to
// expected, including separately stored chunks, or returns a stale-source error.
func (b *BadgerEngine) UpdateNodeEmbeddingIfCurrent(node, expected *Node) error {
	if expected == nil {
		return ErrInvalidData
	}
	return finishEmbeddingUpdate(b.updateEmbeddingDeferred(node, expected))
}

func (b *BadgerEngine) updateEmbeddingDeferred(node, expected *Node) (func(), error) {
	if node == nil {
		return nil, ErrInvalidData
	}
	if node.ID == "" || !strings.Contains(string(node.ID), ":") {
		return nil, ErrInvalidID
	}
	if err := b.ensureOpen(); err != nil {
		return nil, err
	}
	var updated *Node
	err := b.withUpdate(func(txn *badger.Txn) error {
		item, err := txn.Get(nodeKey(node.ID))
		if err == badger.ErrKeyNotFound {
			return ErrNotFound
		}
		if err != nil {
			return err
		}
		var current *Node
		if err := item.Value(func(value []byte) error {
			var err error
			current, err = b.decodeNodeWithEmbeddings(txn, value, node.ID)
			return err
		}); err != nil {
			return err
		}
		if expected != nil {
			same, err := sameEmbeddingSource(current, expected)
			if err != nil {
				return err
			}
			if !same {
				return ErrEmbeddingSourceChanged
			}
		}
		version, err := b.allocateMVCCVersion(txn, namespaceForNodeID(node.ID), time.Now())
		if err != nil {
			return err
		}
		if err := b.archiveNodeOnUpdateInTxn(txn, node.ID); err != nil {
			return err
		}
		updated = CopyNode(current)
		incoming := CopyNode(node)
		updated.ChunkEmbeddings = incoming.ChunkEmbeddings
		updated.EmbedMeta = incoming.EmbedMeta
		updated.UpdatedAt = time.Now()
		data, separate, err := b.encodeNodeInTxn(txn, namespaceForNodeID(node.ID), updated)
		if err != nil {
			return err
		}
		if err := txn.Set(nodeKey(node.ID), data); err != nil {
			return err
		}
		options := badger.DefaultIteratorOptions
		options.Prefix = embeddingPrefix(node.ID)
		iterator := txn.NewIterator(options)
		for iterator.Rewind(); iterator.Valid(); iterator.Next() {
			if err := txn.Delete(iterator.Item().KeyCopy(nil)); err != nil {
				iterator.Close()
				return err
			}
		}
		iterator.Close()
		if separate {
			for i, vector := range updated.ChunkEmbeddings {
				values, err := buildEmbeddingChunkWriteKVs(node.ID, i, vector)
				if err != nil {
					return err
				}
				for _, value := range values {
					if err := txn.Set(value.key, value.val); err != nil {
						return err
					}
				}
			}
		}
		if b.shouldIndexPendingEmbed(updated) {
			if err := txn.Set(pendingEmbedKey(node.ID), nil); err != nil {
				return err
			}
		} else {
			if err := txn.Delete(pendingEmbedKey(node.ID)); err != nil {
				return err
			}
		}
		return b.writeNodeMVCCHeadInTxn(txn, node.ID, version, false)
	})
	if err != nil {
		return nil, err
	}
	return func() { b.cacheOnNodeUpdated(updated); b.notifyNodeUpdated(updated) }, nil
}

func (w *WALEngine) UpdateNodeEmbeddingIfCurrent(node, expected *Node) error {
	if expected == nil {
		return ErrInvalidData
	}
	return finishEmbeddingUpdate(w.updateEmbeddingDeferred(node, expected))
}

func (w *WALEngine) updateEmbeddingDeferred(node, expected *Node) (func(), error) {
	w.mutationMu.RLock()
	defer w.mutationMu.RUnlock()
	if node == nil {
		return nil, ErrInvalidData
	}
	if config.IsWALEnabled() {
		database := w.databaseFromNode(node)
		if err := w.wal.AppendWithDatabase(OpUpdateEmbeddingState, WALNodeData{Node: cloneNodeForWAL(database, node), OldNode: cloneNodeForWAL(database, expected)}, database); err != nil {
			return nil, fmt.Errorf("wal: failed to log embedding update: %w", err)
		}
	}
	return updateEmbeddingDeferred(w.engine, node, expected)
}

func (n *NamespacedEngine) UpdateNodeEmbeddingIfCurrent(node, expected *Node) error {
	if node == nil || expected == nil {
		return ErrInvalidData
	}
	// An external owner such as replication exposes the public contract only.
	if _, local := n.inner.(deferredEmbeddingUpdater); !local {
		updater, ok := n.inner.(ConditionalEmbeddingUpdater)
		if !ok {
			return ErrNotImplemented
		}
		proposed, original := CopyNode(node), CopyNode(expected)
		proposed.ID = n.prefixNodeID(node.ID)
		original.ID = n.prefixNodeID(expected.ID)
		return updater.UpdateNodeEmbeddingIfCurrent(proposed, original)
	}
	return finishEmbeddingUpdate(n.updateEmbeddingDeferred(node, expected))
}

func (n *NamespacedEngine) updateEmbeddingDeferred(node, expected *Node) (func(), error) {
	if node == nil {
		return nil, ErrInvalidData
	}
	proposed := CopyNode(node)
	proposed.ID = n.prefixNodeID(node.ID)
	var original *Node
	if expected != nil {
		original = CopyNode(expected)
		original.ID = n.prefixNodeID(expected.ID)
	}
	return updateEmbeddingDeferred(n.inner, proposed, original)
}

func (ae *AsyncEngine) UpdateNodeEmbeddingIfCurrent(node, expected *Node) error {
	if expected == nil {
		return ErrInvalidData
	}
	return finishEmbeddingUpdate(ae.updateEmbeddingDeferred(node, expected))
}

func (ae *AsyncEngine) updateEmbeddingDeferred(node, expected *Node) (func(), error) {
	if node == nil {
		return nil, ErrInvalidData
	}
	for {
		current, err := ae.GetNode(node.ID)
		if err != nil {
			return nil, err
		}
		if expected != nil {
			same, err := sameEmbeddingSource(current, expected)
			if err != nil {
				return nil, err
			}
			if !same {
				return nil, ErrEmbeddingSourceChanged
			}
		}
		// Persist source writes before recording a provider attempt or accepted result.
		if err := ae.Flush(); err != nil {
			return nil, err
		}
		ae.flushMu.Lock()
		ae.mu.Lock()
		if ae.deleteNodes[node.ID] {
			ae.mu.Unlock()
			ae.flushMu.Unlock()
			return nil, ErrNotFound
		}
		if _, pending := ae.nodeCache[node.ID]; pending || ae.inFlightNodes[node.ID] {
			ae.mu.Unlock()
			ae.flushMu.Unlock()
			continue
		}
		notify, err := updateEmbeddingDeferred(ae.engine, node, expected)
		ae.mu.Unlock()
		ae.flushMu.Unlock()
		return notify, err
	}
}

func (t *TracedEngine) UpdateNodeEmbeddingIfCurrent(node, expected *Node) error {
	if expected == nil {
		return ErrInvalidData
	}
	if _, local := t.Engine.(deferredEmbeddingUpdater); !local {
		updater, ok := t.Engine.(ConditionalEmbeddingUpdater)
		if !ok {
			return ErrNotImplemented
		}
		return updater.UpdateNodeEmbeddingIfCurrent(node, expected)
	}
	return finishEmbeddingUpdate(t.updateEmbeddingDeferred(node, expected))
}

func (t *TracedEngine) updateEmbeddingDeferred(node, expected *Node) (func(), error) {
	return updateEmbeddingDeferred(t.Engine, node, expected)
}

// EmbeddingAttemptSourceCurrent distinguishes a failed/delayed attempt from a
// newly edited source, which starts its own generation attempt budget.
func EmbeddingAttemptSourceCurrent(node *Node) bool {
	recorded, _ := node.EmbedMeta["embedding_attempt_source_fingerprint"].(string)
	if recorded == "" {
		return true
	}
	current, err := EmbeddingSourceFingerprint(node)
	return err == nil && current == recorded
}
