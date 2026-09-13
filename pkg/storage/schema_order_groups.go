package storage

import (
	"fmt"
	"sort"
)

// VisitPropertyIndexGroups visits non-null property-index entries in key order.
// Each callback receives the complete group of IDs whose keys compare equally,
// including differently typed numeric keys. Returning false stops the visit.
// IDs within a group are ordered by node identity. The callback runs without
// schema/index locks so it can read nodes or perform index operations safely.
// The return value reports whether the index exists, even when it is empty.
//
// Query planners use this to finish a primary-key tie before selecting a page
// by secondary ORDER BY keys. For example, a title index must visit every node
// with the boundary title before choosing the smallest timestamp/ID tuple.
func (sm *SchemaManager) VisitPropertyIndexGroups(label, property string, descending bool, visit func([]NodeID) bool) bool {
	sm.mu.RLock()
	idx := sm.propertyIndexes[fmt.Sprintf("%s:%s", label, property)]
	sm.mu.RUnlock()
	if idx == nil {
		return false
	}
	// sortedKeysLocked may refresh its cache, so acquire the write lock.
	idx.mu.Lock()
	keys := idx.sortedKeysLocked()
	idx.mu.Unlock()
	if descending {
		for i, j := 0, len(keys)-1; i < j; i, j = i+1, j-1 {
			keys[i], keys[j] = keys[j], keys[i]
		}
	}
	for start := 0; start < len(keys); {
		end := start + 1
		for end < len(keys) && compareSchemaIndexValues(keys[start], keys[end]) == 0 {
			end++
		}
		var ids []NodeID
		idx.mu.RLock()
		for _, key := range keys[start:end] {
			ids = append(ids, idx.values[key]...)
		}
		idx.mu.RUnlock()
		sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
		if len(ids) > 0 && !visit(ids) {
			break
		}
		start = end
	}
	return true
}
