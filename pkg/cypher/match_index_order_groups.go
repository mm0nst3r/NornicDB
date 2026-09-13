package cypher

import (
	"container/heap"
	"context"
	"sort"
	"strings"

	"github.com/orneryd/nornicdb/pkg/storage"
)

// indexedOrderHeap retains only the requested window; its root is the worst
// candidate under the complete ORDER BY comparator.
type indexedOrderHeap struct {
	nodes   []*storage.Node
	compare func(*storage.Node, *storage.Node) int
}

func (h indexedOrderHeap) Len() int                { return len(h.nodes) }
func (h indexedOrderHeap) Less(i, j int) bool      { return h.compare(h.nodes[i], h.nodes[j]) > 0 }
func (h indexedOrderHeap) Swap(i, j int)           { h.nodes[i], h.nodes[j] = h.nodes[j], h.nodes[i] }
func (h *indexedOrderHeap) Push(value interface{}) { h.nodes = append(h.nodes, value.(*storage.Node)) }
func (h *indexedOrderHeap) Pop() interface{} {
	last := len(h.nodes) - 1
	value := h.nodes[last]
	h.nodes[last] = nil
	h.nodes = h.nodes[:last]
	return value
}

// collectIndexedOrderWindow applies filtering and all sort keys before
// truncation. A primary-key group may be much larger than the requested page;
// only the best limit nodes are retained while every boundary tie is examined.
func (e *StorageExecutor) collectIndexedOrderWindow(ctx context.Context, pattern nodePatternInfo, where string, specs []nodeOrderSpec, label string, limit int) ([]*storage.Node, bool, error) {
	compare := func(a, b *storage.Node) int {
		if cmp := e.compareNodeOrderSpecs(a, b, specs); cmp != 0 {
			return cmp
		}
		return strings.Compare(string(a.ID), string(b.ID))
	}
	top := &indexedOrderHeap{compare: compare}
	var filter func(*storage.Node) bool
	if strings.TrimSpace(where) != "" {
		filter = e.compileNodeWhereFilter(ctx, pattern.variable, where)
	}
	var visitErr error
	visited := false
	found := e.storage.GetSchema().VisitPropertyIndexGroups(label, specs[0].propName, specs[0].descending, func(ids []storage.NodeID) bool {
		visited = true
		for _, id := range ids {
			if err := ctx.Err(); err != nil {
				visitErr = err
				return false
			}
			node, err := e.storage.GetNode(id)
			if err != nil || node == nil {
				continue
			}
			if (len(pattern.labels) > 0 && !nodeHasAnyLabel(node, pattern.labels)) || !e.nodeMatchesProps(node, pattern.properties) {
				continue
			}
			if filter != nil && !filter(node) {
				continue
			}
			if top.Len() < limit {
				heap.Push(top, node)
			} else if compare(node, top.nodes[0]) < 0 {
				top.nodes[0] = node
				heap.Fix(top, 0)
			}
			// With a single key the identity-ordered group already supplies the
			// deterministic tie order; secondary keys require the entire group.
			if len(specs) == 1 && top.Len() == limit {
				return false
			}
		}
		return top.Len() < limit
	})
	if visitErr != nil {
		return nil, false, visitErr
	}
	if !found || !visited {
		return nil, false, nil
	}
	sort.Slice(top.nodes, func(i, j int) bool { return compare(top.nodes[i], top.nodes[j]) < 0 })
	return top.nodes, true, nil
}
