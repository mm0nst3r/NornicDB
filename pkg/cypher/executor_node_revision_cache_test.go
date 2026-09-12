package cypher

import (
	"context"
	"testing"
	"time"

	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

func TestResultCacheTracksStorageNodeRevision(t *testing.T) {
	base, err := storage.NewBadgerEngine(t.TempDir())
	require.NoError(t, err)
	defer base.Close()
	store := storage.NewNamespacedEngine(base, "one")
	other := storage.NewNamespacedEngine(base, "two")
	exec := NewStorageExecutorWithQueryCachePolicy(store, 10, time.Minute)
	const query = "MATCH (n:CacheDelete) RETURN count(n)"
	read := func(want int64) {
		t.Helper()
		result, err := exec.Execute(context.Background(), query, nil)
		require.NoError(t, err)
		require.Equal(t, [][]interface{}{{want}}, result.Rows)
	}
	_, err = store.CreateNode(&storage.Node{ID: "node", Labels: []string{"CacheDelete"}})
	require.NoError(t, err)
	read(1)
	read(1)
	hits, _, _, _, _ := exec.cache.Stats()
	require.Equal(t, int64(1), hits, "unchanged reads must still hit the cache")
	_, err = other.CreateNode(&storage.Node{ID: "other", Labels: []string{"CacheDelete"}})
	require.NoError(t, err)
	read(1)
	hits, _, _, _, _ = exec.cache.Stats()
	require.Equal(t, int64(2), hits, "another database must not invalidate this count")
	require.NoError(t, store.DeleteNode("node"))
	read(0)
	read(0)
	hits, _, _, _, _ = exec.cache.Stats()
	require.Equal(t, int64(3), hits)
}

type pausedLabelCountEngine struct {
	*storage.NamespacedEngine
	counted chan struct{}
	resume  chan struct{}
}

func (s *pausedLabelCountEngine) NodeCountByLabel(label string) (int64, error) {
	count, err := s.NamespacedEngine.NodeCountByLabel(label)
	if s.counted != nil {
		close(s.counted)
		<-s.resume
		s.counted = nil
	}
	return count, err
}

func TestResultCacheReadOverlappingDeleteKeepsOriginalRevision(t *testing.T) {
	base, err := storage.NewBadgerEngine(t.TempDir())
	require.NoError(t, err)
	defer base.Close()
	store := &pausedLabelCountEngine{storage.NewNamespacedEngine(base, "one"), make(chan struct{}), make(chan struct{})}
	_, err = store.CreateNode(&storage.Node{ID: "node", Labels: []string{"CacheDelete"}})
	require.NoError(t, err)
	exec := NewStorageExecutorWithQueryCachePolicy(store, 10, time.Minute)
	const query = "MATCH (n:CacheDelete) RETURN count(n)"
	done := make(chan error, 1)
	go func() {
		_, err := exec.Execute(context.Background(), query, nil)
		done <- err
	}()
	<-store.counted
	err = store.DeleteNode("node")
	close(store.resume)
	require.NoError(t, err)
	require.NoError(t, <-done)
	result, err := exec.Execute(context.Background(), query, nil)
	require.NoError(t, err)
	require.Equal(t, [][]interface{}{{int64(0)}}, result.Rows)
}
