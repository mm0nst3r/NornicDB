package storage

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// chainTestWrapper is an Engine whose GetInnerEngine returns inner, which may
// be the wrapper itself or an outer wrapper, to build self-returning and
// cyclic chains.
type chainTestWrapper struct {
	Engine
	name  string
	inner Engine
}

func (w *chainTestWrapper) GetInnerEngine() Engine { return w.inner }

func chainNames(engine Engine) []string {
	var names []string
	for layer := range EngineChain(engine) {
		if wrapper, ok := layer.(*chainTestWrapper); ok {
			names = append(names, wrapper.name)
			continue
		}
		names = append(names, "base")
	}
	return names
}

// TestEngineChain covers #691: a wrapper whose GetInnerEngine returns itself
// (CompositeEngine does) or a chain that leads back to an earlier engine ends
// the walk instead of looping forever.
func TestEngineChain(t *testing.T) {
	base := NewMemoryEngine()
	t.Cleanup(func() { _ = base.Close() })

	outer := &chainTestWrapper{name: "outer", inner: &chainTestWrapper{name: "middle", inner: base}}
	require.Equal(t, []string{"outer", "middle", "base"}, chainNames(outer))

	self := &chainTestWrapper{name: "self"}
	self.inner = self
	require.Equal(t, []string{"self"}, chainNames(self))
	require.Nil(t, InnerEngine(self))

	first := &chainTestWrapper{name: "first"}
	second := &chainTestWrapper{name: "second", inner: first}
	first.inner = second
	require.Equal(t, []string{"first", "second"}, chainNames(first))

	composite := NewCompositeEngine(map[string]Engine{"a": base}, map[string]string{"a": "a"}, map[string]string{"a": "read_write"})
	require.Len(t, chainNames(composite), 1)

	require.Empty(t, chainNames(nil))

	allocs := testing.AllocsPerRun(100, func() {
		for layer := range EngineChain(outer) {
			_ = layer
		}
	})
	require.Zero(t, allocs)
}
