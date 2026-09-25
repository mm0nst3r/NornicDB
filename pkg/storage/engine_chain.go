package storage

import "iter"

// maxEngineChainDepth bounds EngineChain. Production wrapper chains
// (size-tracking → namespaced → async → WAL → base, plus replication and
// transaction wrappers) are well under it.
const maxEngineChainDepth = 16

// InnerEngine returns the engine that engine wraps, from its GetInnerEngine,
// GetEngine or GetUnderlying method (a wrapper's methods all return the same
// inner engine). It returns nil when engine wraps nothing, and when the method
// returns engine itself: CompositeEngine.GetInnerEngine does, because a
// composite has no single inner engine.
func InnerEngine(engine Engine) Engine {
	var inner Engine
	switch wrapper := engine.(type) {
	case interface{ GetInnerEngine() Engine }:
		inner = wrapper.GetInnerEngine()
	case interface{ GetEngine() Engine }:
		inner = wrapper.GetEngine()
	case interface{ GetUnderlying() Engine }:
		inner = wrapper.GetUnderlying()
	default:
		return nil
	}
	if inner == engine {
		return nil
	}
	return inner
}

// EngineChain yields engine and then every engine under it, outermost first,
// following InnerEngine down to the base engine. It never yields an engine
// twice, so a wrapper that returns itself or a chain that leads back to an
// engine already seen ends the walk instead of looping forever (#691). Every
// walk over a wrapper chain goes through it. It doesn't allocate.
func EngineChain(engine Engine) iter.Seq[Engine] {
	return func(yield func(Engine) bool) {
		var seen [maxEngineChainDepth]Engine
		current := engine
		for depth := 0; current != nil && depth < len(seen); depth++ {
			for _, previous := range seen[:depth] {
				if previous == current {
					return
				}
			}
			seen[depth] = current
			if !yield(current) {
				return
			}
			current = InnerEngine(current)
		}
	}
}
