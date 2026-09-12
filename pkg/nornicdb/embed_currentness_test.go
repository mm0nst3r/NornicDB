package nornicdb

import (
	"context"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
	"testing"
	"time"
)

type heldEmbeddingProvider struct {
	entered chan struct{}
	release chan struct{}
}

func (p *heldEmbeddingProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	v, e := p.EmbedBatch(ctx, []string{text})
	if e != nil {
		return nil, e
	}
	return v[0], nil
}
func (p *heldEmbeddingProvider) EmbedBatch(ctx context.Context, text []string) ([][]float32, error) {
	close(p.entered)
	select {
	case <-p.release:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	out := make([][]float32, len(text))
	for i := range out {
		out[i] = []float32{1, 0}
	}
	return out, nil
}
func (p *heldEmbeddingProvider) ChunkText(text string, _, _ int) ([]string, error) {
	return []string{text}, nil
}
func (p *heldEmbeddingProvider) Model() string   { return "synthetic-held" }
func (p *heldEmbeddingProvider) Dimensions() int { return 2 }
func (p *heldEmbeddingProvider) Backend() string { return "cpu" }

func TestManagedEmbeddingRejectsOrdinaryInFlightContentEdit(t *testing.T) {
	engine := storage.NewMemoryEngine()
	defer engine.Close()
	engine.SetEmbeddingsEnabled(true)
	_, err := engine.CreateNode(&storage.Node{ID: "test:document", Labels: []string{"Document"}, Properties: map[string]any{"content": "old source"}})
	require.NoError(t, err)
	provider := &heldEmbeddingProvider{entered: make(chan struct{}), release: make(chan struct{})}
	cfg := DefaultEmbedWorkerConfig()
	cfg.DeferWorkerStart = true
	cfg.BatchDelay = 0
	worker := NewEmbedWorker(provider, engine, cfg)
	defer worker.Close()
	finished := make(chan struct{})
	go func() { worker.processNextBatch(); close(finished) }()
	select {
	case <-provider.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("generation did not start")
	}
	current, err := engine.GetNode("test:document")
	require.NoError(t, err)
	current.Properties["content"] = "new source"
	require.NoError(t, engine.UpdateNode(current))
	close(provider.release)
	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("generation did not finish")
	}
	current, err = engine.GetNode("test:document")
	require.NoError(t, err)
	require.Equal(t, "new source", current.Properties["content"])
	require.Empty(t, current.ChunkEmbeddings, "old-source vectors must not attach to the edited document")
}
