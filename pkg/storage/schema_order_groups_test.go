package storage

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestVisitPropertyIndexGroups(t *testing.T) {
	sm := NewSchemaManager()
	called := false
	require.False(t, sm.VisitPropertyIndexGroups("Item", "rank", false, func([]NodeID) bool {
		called = true
		return true
	}))
	require.False(t, called)
	require.NoError(t, sm.AddPropertyIndex("rank", "Item", []string{"rank"}))
	require.True(t, sm.VisitPropertyIndexGroups("Item", "rank", false, func([]NodeID) bool {
		called = true
		return true
	}))
	require.False(t, called)
	for _, entry := range []struct {
		id    NodeID
		value interface{}
	}{
		{"b", int64(1)}, {"a", float64(1)}, {"c", int64(2)}, {"null", nil},
	} {
		require.NoError(t, sm.PropertyIndexInsert("Item", "rank", entry.id, entry.value))
	}
	for _, descending := range []bool{false, true} {
		var groups [][]NodeID
		require.True(t, sm.VisitPropertyIndexGroups("Item", "rank", descending, func(ids []NodeID) bool {
			groups = append(groups, ids)
			// The callback can safely use the index API; no index lock is held.
			require.Equal(t, []NodeID{"c"}, sm.PropertyIndexLookup("Item", "rank", int64(2)))
			return true
		}))
		expected := [][]NodeID{{"a", "b"}, {"c"}}
		if descending {
			expected = [][]NodeID{{"c"}, {"a", "b"}}
		}
		require.Equal(t, expected, groups)
	}
	visits := 0
	require.True(t, sm.VisitPropertyIndexGroups("Item", "rank", false, func(ids []NodeID) bool {
		visits++
		require.Equal(t, []NodeID{"a", "b"}, ids)
		return false
	}))
	require.Equal(t, 1, visits)
}
