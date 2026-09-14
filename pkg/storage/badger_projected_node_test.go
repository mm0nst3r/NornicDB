package storage

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBadgerEngine_GetNodeProjectedSkipsUnrequestedVectorProperty(t *testing.T) {
	engine := createTestBadgerEngine(t)
	embedding := make([]float64, 1024)
	for i := range embedding {
		embedding[i] = float64(i) / 1024
	}
	node := testNode("projected-node")
	node.Labels = []string{"Entity"}
	node.Properties = map[string]any{
		"uuid":           "entity-1",
		"group_id":       "episode-1",
		"name_embedding": embedding,
	}

	_, err := engine.CreateNode(node)
	require.NoError(t, err)

	projected, err := engine.GetNodeProjected(node.ID, []string{"uuid", "group_id"})
	require.NoError(t, err)
	require.Equal(t, node.ID, projected.ID)
	require.Equal(t, []string{"Entity"}, projected.Labels)
	require.Equal(t, "entity-1", projected.Properties["uuid"])
	require.Equal(t, "episode-1", projected.Properties["group_id"])
	require.NotContains(t, projected.Properties, "name_embedding")

	full, err := engine.GetNode(node.ID)
	require.NoError(t, err)
	require.Contains(t, full.Properties, "name_embedding")
	require.Len(t, full.Properties["name_embedding"], 1024)
}

func TestBadgerEngine_GetNodeProjectedEmptyPropertyList(t *testing.T) {
	engine := createTestBadgerEngine(t)
	node := testNode("projected-empty")
	node.Properties = map[string]any{"uuid": "entity-1"}

	_, err := engine.CreateNode(node)
	require.NoError(t, err)

	projected, err := engine.GetNodeProjected(node.ID, []string{})
	require.NoError(t, err)
	require.Equal(t, node.ID, projected.ID)
	require.Empty(t, projected.Properties)
}

func TestNamespacedEngine_GetNodeWithoutEmbeddingsSkipsSeparateVectors(t *testing.T) {
	engine := createTestBadgerEngine(t)
	tenant := NewNamespacedEngine(engine, "tenant")
	node := &Node{
		ID:         "large-vector-node",
		Labels:     []string{"Evidence"},
		Properties: map[string]any{"asset_id": "asset-a"},
		ChunkEmbeddings: [][]float32{
			make([]float32, 10_000),
			make([]float32, 10_000),
		},
	}
	_, err := tenant.CreateNode(node)
	require.NoError(t, err)

	light, err := tenant.GetNodeWithoutEmbeddings(node.ID)
	require.NoError(t, err)
	require.Equal(t, node.ID, light.ID)
	require.Equal(t, "asset-a", light.Properties["asset_id"])
	require.Empty(t, light.ChunkEmbeddings)
	require.Empty(t, light.NamedEmbeddings)

	full, err := tenant.GetNode(node.ID)
	require.NoError(t, err)
	require.False(t, full.EmbeddingsStoredSeparately)
	require.Len(t, full.ChunkEmbeddings, 2)
	require.Len(t, full.ChunkEmbeddings[0], 10_000)
}

func TestNamespacedEngine_BatchGetNodesWithoutEmbeddingsSkipsSeparateVectors(t *testing.T) {
	engine := createTestBadgerEngine(t)
	tenant := NewNamespacedEngine(engine, "tenant")
	for _, id := range []NodeID{"large-vector-a", "large-vector-b"} {
		node := &Node{
			ID:         id,
			Labels:     []string{"Evidence"},
			Properties: map[string]any{"asset_id": string(id) + "-asset"},
			ChunkEmbeddings: [][]float32{
				make([]float32, 10_000),
				make([]float32, 10_000),
			},
		}
		_, err := tenant.CreateNode(node)
		require.NoError(t, err)
	}

	light, err := tenant.BatchGetNodesWithoutEmbeddings([]NodeID{"large-vector-a", "missing", "large-vector-b"})
	require.NoError(t, err)
	require.Len(t, light, 2)
	require.Equal(t, NodeID("large-vector-a"), light["large-vector-a"].ID)
	require.Equal(t, "large-vector-a-asset", light["large-vector-a"].Properties["asset_id"])
	require.Empty(t, light["large-vector-a"].ChunkEmbeddings)
	require.Empty(t, light["large-vector-b"].ChunkEmbeddings)
}

func TestNamespacedEngine_StreamNodesByLabelProjected(t *testing.T) {
	engine := createTestBadgerEngine(t)
	tenantA := NewNamespacedEngine(engine, "tenant_a")
	tenantB := NewNamespacedEngine(engine, "tenant_b")

	_, err := tenantA.CreateNode(&Node{
		ID:     "one",
		Labels: []string{"Evidence"},
		Properties: map[string]any{
			"asset_id":  "wanted",
			"embedding": make([]float64, 1024),
		},
	})
	require.NoError(t, err)
	_, err = tenantB.CreateNode(&Node{
		ID:     "two",
		Labels: []string{"Evidence"},
		Properties: map[string]any{
			"asset_id": "other",
		},
	})
	require.NoError(t, err)

	var nodes []*Node
	err = tenantA.StreamNodesByLabelProjected("Evidence", []string{"asset_id"}, func(node *Node) error {
		nodes = append(nodes, node)
		return nil
	})
	require.NoError(t, err)
	require.Len(t, nodes, 1)
	require.Equal(t, NodeID("one"), nodes[0].ID)
	require.Equal(t, "wanted", nodes[0].Properties["asset_id"])
	require.NotContains(t, nodes[0].Properties, "embedding")
}

func TestNamespacedEngine_StreamNodesByPrefixProjected(t *testing.T) {
	engine := createTestBadgerEngine(t)
	tenantA := NewNamespacedEngine(engine, "tenant_a")
	tenantB := NewNamespacedEngine(engine, "tenant_b")

	_, err := tenantA.CreateNode(&Node{
		ID:     "evidence-one",
		Labels: []string{"Evidence"},
		Properties: map[string]any{
			"asset_id":  "wanted",
			"embedding": make([]float64, 1024),
		},
	})
	require.NoError(t, err)
	_, err = tenantA.CreateNode(&Node{
		ID:         "other-one",
		Labels:     []string{"Evidence"},
		Properties: map[string]any{"asset_id": "same-tenant-other-prefix"},
	})
	require.NoError(t, err)
	_, err = tenantB.CreateNode(&Node{
		ID:         "evidence-two",
		Labels:     []string{"Evidence"},
		Properties: map[string]any{"asset_id": "other-tenant"},
	})
	require.NoError(t, err)

	var nodes []*Node
	err = tenantA.StreamNodesByPrefixProjected(context.Background(), "evidence", []string{"asset_id"}, func(node *Node) error {
		nodes = append(nodes, node)
		return nil
	})
	require.NoError(t, err)
	require.Len(t, nodes, 1)
	require.Equal(t, NodeID("evidence-one"), nodes[0].ID)
	require.Equal(t, "wanted", nodes[0].Properties["asset_id"])
	require.NotContains(t, nodes[0].Properties, "embedding")
}
