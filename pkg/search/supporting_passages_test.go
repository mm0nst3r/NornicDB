package search

import (
	"context"
	"strings"
	"testing"

	"github.com/orneryd/nornicdb/pkg/resultstream"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

// Search enrichment reads nodes without their stored vectors. A current
// contextualized document must still expose its provider passages through
// that projection on every engine that offers it.
func TestSupportingPassagesSurviveProjectedNodeReads(t *testing.T) {
	for _, kind := range []string{"memory", "badger"} {
		t.Run(kind, func(t *testing.T) {
			var base storage.Engine
			if kind == "badger" {
				engine, err := storage.NewBadgerEngineInMemory()
				require.NoError(t, err)
				base = engine
			} else {
				base = storage.NewMemoryEngine()
			}
			t.Cleanup(func() { require.NoError(t, base.Close()) })
			engine := storage.NewNamespacedEngine(base, "review")

			node := &storage.Node{
				ID:              "doc",
				Labels:          []string{"Document"},
				Properties:      map[string]any{"content": "First exact passage. Second exact passage."},
				ChunkEmbeddings: [][]float32{{1, 0}, {0, 1}},
				EmbedMeta: map[string]any{
					"embedding_space": "review-space",
					"embedding_api":   "contextualizedembeddings",
					"chunk_texts":     []string{"First exact passage.", "Second exact passage."},
				},
			}
			fingerprint, err := storage.EmbeddingSourceFingerprint(node)
			require.NoError(t, err)
			node.EmbedMeta["embedding_source_fingerprint"] = fingerprint
			_, err = engine.CreateNode(node)
			require.NoError(t, err)

			service := NewServiceWithDimensions(engine, 2)
			t.Cleanup(func() { require.NoError(t, service.Close()) })
			service.SetEmbeddingSpace("review-space")
			require.NoError(t, service.BuildIndexes(context.Background()))

			light, err := service.getNodeWithoutEmbeddings("doc")
			require.NoError(t, err)
			require.Empty(t, light.ChunkEmbeddings, "the projection under test must omit vectors")

			zero := 0.0
			opts := DefaultSearchOptions()
			opts.Limit = 10
			opts.MinSimilarity = &zero
			response, err := service.Search(context.Background(), "Second", []float32{0, 1}, opts)
			require.NoError(t, err)
			require.Len(t, response.Results, 1)
			passages := response.Results[0].SupportingPassages
			require.NotEmpty(t, passages, "current provider chunks must survive projected enrichment")
			for _, passage := range passages {
				require.Equal(t, "doc", passage.NodeID)
				require.Equal(t, fingerprint, passage.SourceFingerprint)
				require.Equal(t, "review-space", passage.Space)
			}

			// A source edit after generation invalidates the stored chunks. Storage
			// events re-index the edited node in production; do the same here. The
			// edited text still matches lexically, so the hit remains and only its
			// stale provider passages must disappear.
			edited, err := engine.GetNode("doc")
			require.NoError(t, err)
			edited.Properties["content"] = "Second passage edited after embedding."
			require.NoError(t, engine.UpdateNode(edited))
			require.NoError(t, service.IndexNode(edited))
			response, err = service.Search(context.Background(), "Second", []float32{0, 1}, opts)
			require.NoError(t, err)
			require.Len(t, response.Results, 1)
			require.Empty(t, response.Results[0].SupportingPassages, "stale chunks must not be exposed")
		})
	}
}

// Provider passages survive cursor compaction, so the shared retained-byte
// estimate must count their text and the registry limits must apply to it.
func TestContinuationRetainsSupportingPassageBytes(t *testing.T) {
	t.Run("accounting", func(t *testing.T) {
		text := strings.Repeat("source passage text ", 50000)
		compact := compactContinuationResult(SearchResult{
			ID: "doc", NodeID: "doc",
			SupportingPassages: []SupportingPassage{{NodeID: "doc", Text: text, MatchedBy: []string{"vector", "bm25"}, Space: "s"}},
		})
		require.Len(t, compact.SupportingPassages, 1, "compaction keeps provider passages")
		require.GreaterOrEqual(t, compactContinuationResultBytes(compact), int64(len(text)))
	})

	t.Run("registry admission", func(t *testing.T) {
		base := storage.NewMemoryEngine()
		t.Cleanup(func() { require.NoError(t, base.Close()) })
		engine := storage.NewNamespacedEngine(base, "review")
		service := NewServiceWithDimensions(engine, 2)
		t.Cleanup(func() { require.NoError(t, service.Close()) })
		registry, err := resultstream.NewRegistry(resultstream.Config{MaxRetainedBytes: 32000, MaxRetainedBytesPerOwner: 32000})
		require.NoError(t, err)
		t.Cleanup(registry.Close)
		service.SetContinuationRegistry(registry)
		for _, id := range []storage.NodeID{"a", "b"} {
			_, err := engine.CreateNode(&storage.Node{ID: id, Properties: map[string]any{"content": "source"}})
			require.NoError(t, err)
		}
		text := strings.Repeat("source text ", 3000)
		producer := func(context.Context, string, []float32, *SearchOptions) (*SearchResponse, error) {
			return &SearchResponse{Status: "success", RetrievalExhausted: true, Results: []SearchResult{
				{ID: "a", NodeID: "a", SupportingPassages: []SupportingPassage{{NodeID: "a", Text: text}}},
				{ID: "b", NodeID: "b", SupportingPassages: []SupportingPassage{{NodeID: "b", Text: text + " two"}}},
			}}, nil
		}
		opts := DefaultSearchOptions()
		opts.Limit = 2
		for _, mode := range []SearchContinuationMode{SearchContinuationRanked, SearchContinuationRankedThenID} {
			request := SearchContinuationRequest{Owner: "reviewer", Database: "review", N: 1, Mode: mode, RankedLimit: 2}
			_, err := service.SearchTextContinuation(context.Background(), "source", opts, request, nil, nil, producer, ChunkedSearchErrorPolicy{})
			require.ErrorIs(t, err, resultstream.ErrCapacity, "mode %s must count retained provider text against the cursor limit", mode)
		}
	})
}
