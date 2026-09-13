package search

import (
	"errors"
	"github.com/orneryd/nornicdb/pkg/storage"
)

// SetEmbeddingSpace selects the database's native model space. A matching number
// of dimensions is insufficient. Empty selects ordinary, non-native vectors.
// Configure before building indexes; a model change requires rebuilding indexes.
func (s *Service) SetEmbeddingSpace(space string) {
	if previous, ok := s.embeddingSpace.Load().(string); ok && previous == space {
		return
	}
	s.embeddingSpace.Store(space)
	if s.resultCache != nil {
		s.resultCache.Invalidate()
	}
}

func (s *Service) embeddingNodeEligible(node *storage.Node) bool {
	actual, _ := node.EmbedMeta["embedding_space"].(string)
	expected, _ := s.embeddingSpace.Load().(string)
	if actual != expected {
		return false
	}
	if actual != "" && !storage.ManagedEmbeddingCurrent(node) {
		return false
	}
	return true
}

func (s *Service) filterCurrentEmbeddingResults(rows []indexResult) ([]indexResult, error) {
	if s.embeddingSpace.Load() == nil {
		return rows, nil
	}
	out := make([]indexResult, 0, len(rows))
	for _, row := range rows {
		node, err := s.engine.GetNode(storage.NodeID(row.ID))
		if errors.Is(err, storage.ErrNotFound) {
			expected, _ := s.embeddingSpace.Load().(string)
			if expected == "" {
				out = append(out, row)
			}
			continue
		}
		if err != nil {
			return nil, err
		}
		if node != nil && s.embeddingNodeEligible(node) {
			out = append(out, row)
		}
	}
	return out, nil
}
