package storage

import (
	"strings"
	"sync"
)

// NodeMutationVersionProvider reports a process-local revision of visible node
// data. A changed revision invalidates a retained search population; it is not a
// transaction timestamp or a durable snapshot. supported is false when an
// underlying external engine cannot report its mutations.
type NodeMutationVersionProvider interface {
	NodeMutationVersion() (version uint64, supported bool)
}

// NamespaceNodeMutationVersionProvider scopes node revisions to one database.
type NamespaceNodeMutationVersionProvider interface {
	NodeMutationVersionInNamespace(namespace string) (version uint64, supported bool)
}

// nodeMutationVersions tracks committed storage publication separately from
// search indexing. In AsyncEngine it tracks newly visible buffered changes too.
type nodeMutationVersions struct {
	mu          sync.RWMutex
	all         uint64
	byNamespace map[string]uint64
}

func (v *nodeMutationVersions) changed(id NodeID) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.all++
	if v.byNamespace == nil {
		v.byNamespace = make(map[string]uint64)
	}
	v.byNamespace[namespaceForNodeID(id)]++
}

func (v *nodeMutationVersions) read(namespace string) uint64 {
	v.mu.Lock()
	defer v.mu.Unlock()
	if namespace == "" {
		return v.all
	}
	if v.byNamespace == nil {
		v.byNamespace = make(map[string]uint64)
	}
	if _, exists := v.byNamespace[namespace]; !exists {
		v.byNamespace[namespace] = 0
	}
	return v.byNamespace[namespace]
}

func (v *nodeMutationVersions) changedPrefix(prefix string) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.all++
	for namespace := range v.byNamespace {
		if strings.HasPrefix(namespace+":", prefix) || strings.HasPrefix(prefix, namespace+":") {
			v.byNamespace[namespace]++
		}
	}
}

// NodeMutationVersion reports all node publications through this engine.
func (b *BadgerEngine) NodeMutationVersion() (uint64, bool) {
	return b.nodeMutationVersions.read(""), true
}

// NodeMutationVersionInNamespace reports only this database's publications.
func (b *BadgerEngine) NodeMutationVersionInNamespace(namespace string) (uint64, bool) {
	return b.nodeMutationVersions.read(namespace), true
}

func nodeVersion(engine Engine, namespace string) (uint64, bool) {
	if namespace != "" {
		if provider, ok := engine.(NamespaceNodeMutationVersionProvider); ok {
			return provider.NodeMutationVersionInNamespace(namespace)
		}
		return 0, false
	}
	if provider, ok := engine.(NodeMutationVersionProvider); ok {
		return provider.NodeMutationVersion()
	}
	return 0, false
}

// NodeMutationVersion reports changes visible through this namespace.
func (n *NamespacedEngine) NodeMutationVersion() (uint64, bool) {
	return nodeVersion(n.inner, n.namespace)
}

// NodeMutationVersionInNamespace keeps nested views in their containing physical
// database's revision scope. Stored IDs are partitioned by their outer prefix.
func (n *NamespacedEngine) NodeMutationVersionInNamespace(namespace string) (uint64, bool) {
	return n.NodeMutationVersion()
}

// NodeMutationVersion includes both buffered writes and underlying publications.
func (ae *AsyncEngine) NodeMutationVersion() (uint64, bool) {
	return ae.NodeMutationVersionInNamespace("")
}

// NodeMutationVersionInNamespace includes this database's buffered and persisted changes.
func (ae *AsyncEngine) NodeMutationVersionInNamespace(namespace string) (uint64, bool) {
	inner, ok := nodeVersion(ae.engine, namespace)
	return inner + ae.nodeMutationVersions.read(namespace), ok
}

// NodeMutationVersion forwards the underlying engine's publication revision.
func (w *WALEngine) NodeMutationVersion() (uint64, bool) { return nodeVersion(w.engine, "") }

// NodeMutationVersionInNamespace forwards a database-scoped publication revision.
func (w *WALEngine) NodeMutationVersionInNamespace(namespace string) (uint64, bool) {
	return nodeVersion(w.engine, namespace)
}

// NodeMutationVersion preserves revision reporting through tracing wrappers.
func (t *TracedEngine) NodeMutationVersion() (uint64, bool) { return nodeVersion(t.Engine, "") }

// NodeMutationVersionInNamespace preserves database-scoped revision reporting.
func (t *TracedEngine) NodeMutationVersionInNamespace(namespace string) (uint64, bool) {
	return nodeVersion(t.Engine, namespace)
}
