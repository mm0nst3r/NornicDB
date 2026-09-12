package storage

import (
	"errors"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestAsyncEngine_UpdateDeleteOverlappingFlush(t *testing.T) {
	previous := runtime.GOMAXPROCS(2)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })
	base, err := NewBadgerEngine(t.TempDir())
	require.NoError(t, err)
	gate := &asyncFlushReturnGate{Engine: base}
	ae := NewAsyncEngine(gate, &AsyncEngineConfig{FlushInterval: time.Hour})
	t.Cleanup(func() { require.NoError(t, ae.Close()) })
	id := NodeID("test:flush-overlap")
	// Repeated client edits overlap the supported flush worker. Each round
	// checks deletion before the next flush can hide a classification mistake.
	for round := 0; round < 2000; round++ {
		_, err := base.CreateNode(&Node{ID: id, Properties: map[string]any{"value": "original"}})
		require.NoError(t, err)
		require.NoError(t, ae.UpdateNode(&Node{ID: id, Properties: map[string]any{"value": "first"}}))
		start := make(chan struct{})
		gate.persisted, gate.release = make(chan struct{}), make(chan struct{})
		flushed, updated := make(chan error, 1), make(chan error, 1)
		go func() { flushed <- ae.Flush() }()
		<-gate.persisted
		go func() {
			<-start
			for edit := 0; edit < 32; edit++ {
				if err := ae.UpdateNode(&Node{ID: id, Properties: map[string]any{"value": edit}}); err != nil {
					updated <- err
					return
				}
			}
			updated <- nil
		}()
		close(start)
		close(gate.release)
		require.NoError(t, <-updated)
		require.NoError(t, <-flushed)
		require.NoError(t, ae.DeleteNode(id))
		_, err = ae.GetNode(id)
		require.ErrorIs(t, err, ErrNotFound, "round %d: delete after overlapping update/flush must hide the stored row", round)
		require.NoError(t, ae.Flush())
		_, err = base.GetNode(id)
		require.ErrorIs(t, err, ErrNotFound, "round %d: deletion must persist", round)
	}
}

// Delay only the return from a real successful write so cleanup overlaps a
// client edit burst, rather than spending the overlap window in disk I/O.
type asyncFlushReturnGate struct {
	Engine
	persisted chan struct{}
	release   chan struct{}
}

func (e *asyncFlushReturnGate) UpdateNode(node *Node) error {
	err := e.Engine.UpdateNode(node)
	close(e.persisted)
	<-e.release
	return err
}

type firstAsyncWriteGate struct {
	Engine
	started chan struct{}
	release chan struct{}
	once    sync.Once
	err     error
}

func (e *firstAsyncWriteGate) UpdateNode(node *Node) error {
	first := false
	e.once.Do(func() {
		first = true
		close(e.started)
		<-e.release
	})
	if first && e.err != nil {
		return e.err
	}
	return e.Engine.UpdateNode(node)
}

func TestAsyncEngine_UpdateDuringFirstFlushClassification(t *testing.T) {
	for _, fail := range []bool{false, true} {
		name := "successful-first-write"
		if fail {
			name = "failed-first-write"
		}
		t.Run(name, func(t *testing.T) {
			base, err := NewBadgerEngine(t.TempDir())
			require.NoError(t, err)
			gate := &firstAsyncWriteGate{Engine: base, started: make(chan struct{}), release: make(chan struct{})}
			if fail {
				// Fault injection models one rejected write before real Badger
				// persistence; the normal retry/cancellation queue stays active.
				gate.err = errors.New("injected backing-store write failure")
			}
			ae := NewAsyncEngine(gate, &AsyncEngineConfig{FlushInterval: time.Hour})
			t.Cleanup(func() { require.NoError(t, ae.Close()) })
			id := NodeID("test:first-flush")
			_, err = ae.CreateNode(&Node{ID: id, Properties: map[string]any{"value": "first"}})
			require.NoError(t, err)
			flushed := make(chan error, 1)
			go func() { flushed <- ae.Flush() }()
			select {
			case <-gate.started:
			case <-time.After(5 * time.Second):
				t.Fatal("flush did not reach backing store")
			}
			// A different object ensures a newer queued write survives the
			// cleanup of the first snapshot.
			err = ae.UpdateNode(&Node{ID: id, Properties: map[string]any{"value": "second"}})
			close(gate.release)
			require.NoError(t, err)
			if fail {
				require.Error(t, <-flushed)
			} else {
				require.NoError(t, <-flushed)
			}
			count, err := ae.NodeCount()
			require.NoError(t, err)
			require.EqualValues(t, 1, count)
			count, err = ae.NodeCountByPrefix("test:")
			require.NoError(t, err)
			require.EqualValues(t, 1, count)
			if fail {
				_, err = base.GetNode(id)
				require.ErrorIs(t, err, ErrNotFound)
			}
			require.NoError(t, ae.DeleteNode(id))
			_, err = ae.GetNode(id)
			require.ErrorIs(t, err, ErrNotFound)
			require.NoError(t, ae.Flush())
			_, err = base.GetNode(id)
			require.ErrorIs(t, err, ErrNotFound)
		})
	}
}

type asyncLookupErrorEngine struct {
	Engine
	err error
}

func (e *asyncLookupErrorEngine) GetNode(id NodeID) (*Node, error) {
	return nil, e.err
}

func TestAsyncEngine_UpdateNodeLookupFailure(t *testing.T) {
	base, err := NewBadgerEngine(t.TempDir())
	require.NoError(t, err)
	id := NodeID("test:lookup-failure")
	_, err = base.CreateNode(&Node{ID: id, Properties: map[string]any{"value": "original"}})
	require.NoError(t, err)
	lookupErr := errors.New("injected backing-store read failure")
	ae := NewAsyncEngine(&asyncLookupErrorEngine{Engine: base, err: lookupErr}, &AsyncEngineConfig{FlushInterval: time.Hour})
	t.Cleanup(func() { _ = ae.Close() })
	err = ae.UpdateNode(&Node{ID: id, Properties: map[string]any{"value": "edited"}})
	require.ErrorIs(t, err, lookupErr)
	require.NoError(t, ae.Flush(), "a rejected lookup must not queue a write")
	stored, err := base.GetNode(id)
	require.NoError(t, err)
	require.Equal(t, "original", stored.Properties["value"])
}
