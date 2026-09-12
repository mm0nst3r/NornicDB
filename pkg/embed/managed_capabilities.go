package embed

import (
	"context"
	"fmt"
)

// ManagedProvider unwraps the existing cache without applying its text-only cache
// to whole documents, chunk context, or image inputs. Other embedders keep their
// existing worker behavior. Wrappers outside CachedEmbedder must explicitly retain
// this capability rather than masquerading as native managed providers.
func ManagedProvider(e any) (ManagedDocumentEmbedder, bool) {
	e = unwrapProvider(e)
	provider, ok := e.(ManagedDocumentEmbedder)
	return provider, ok
}

func unwrapProvider(e any) any {
	for {
		switch wrapper := e.(type) {
		case *CachedEmbedder:
			if wrapper == nil {
				return nil
			}
			e = wrapper.base
		case *TracedEmbedder:
			if wrapper == nil {
				return nil
			}
			e = wrapper.inner
		default:
			return e
		}
	}
}

// QueryProvider resolves explicit query purpose through supported decorators.
// A decorator's forwarding method must not make a legacy provider appear native.
func QueryProvider(e any) (interface {
	EmbedQuery(context.Context, string) ([]float32, error)
}, bool) {
	provider, ok := unwrapProvider(e).(interface {
		EmbedQuery(context.Context, string) ([]float32, error)
	})
	return provider, ok
}

// QueryVector uses an explicit query capability when available; legacy providers
// retain their original Embed behavior. The query owner must use this helper (or
// the equivalent small interface assertion in layers avoiding an embed import).
func QueryVector(ctx context.Context, e interface {
	Embed(context.Context, string) ([]float32, error)
}, text string) ([]float32, error) {
	if e == nil {
		return nil, fmt.Errorf("query embedder is required")
	}
	if q, ok := QueryProvider(e); ok {
		return q.EmbedQuery(ctx, text)
	}

	return e.Embed(ctx, text)
}

// EmbedQuery retains a native query capability through CachedEmbedder. Native
// queries bypass its shared document/text cache, avoiding purpose collisions.
// Legacy queries keep the existing cache. No native document uses this cache.
func (c *CachedEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	if c == nil || c.base == nil {
		return nil, fmt.Errorf("query embedder is required")
	}
	if q, ok := QueryProvider(c.base); ok {
		return q.EmbedQuery(ctx, text)
	}
	return c.Embed(ctx, text)
}

// NativeEmbeddingSpace returns the explicit native provider identity, if any.
func NativeEmbeddingSpace(provider any) string {
	if managed, ok := ManagedProvider(provider); ok {
		return managed.EmbeddingSpace().Key()
	}
	return ""
}
