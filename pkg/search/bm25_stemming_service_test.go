package search

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
)

type stemmingRebuildIndex struct {
	bm25Index
	indexCalls int
}

func (i *stemmingRebuildIndex) IndexBatch(entries []FulltextBatchEntry) {
	i.indexCalls++
	i.bm25Index.IndexBatch(entries)
}

func (i *stemmingRebuildIndex) AnalyzerIdentity() string {
	return i.bm25Index.(interface{ AnalyzerIdentity() string }).AnalyzerIdentity()
}

// Exercise the owner of snapshot loading/rebuilding, not a simulated refill of
// an index after Load. The same-language reopen must actually reuse its snapshot.
func TestBM25StemmingServiceRebuild(t *testing.T) {
	for _, engine := range []string{BM25EngineV1, BM25EngineV2} {
		t.Run(engine, func(t *testing.T) {
			root := storage.NewMemoryEngine()
			t.Cleanup(func() { require.NoError(t, root.Close()) })
			store := storage.NewNamespacedEngine(root, "test")
			_, err := store.CreateNode(&storage.Node{
				ID: "wagon", Labels: []string{"Doc"},
				Properties: map[string]any{"text": "вагон Ёлка running"},
			})
			require.NoError(t, err)
			path := filepath.Join(t.TempDir(), "bm25")
			for _, phase := range []struct {
				language string
				reuse    bool
			}{
				{"none", false}, {"russian", false}, {"russian", true},
				{"english", false}, {"none", false},
			} {
				t.Run(phase.language, func(t *testing.T) {
					t.Setenv(EnvBM25Stemmer, phase.language)
					svc := NewServiceWithDimensionsAndBM25Engine(store, 3, engine)
					index := &stemmingRebuildIndex{bm25Index: svc.fulltextIndex}
					svc.fulltextIndex = index
					t.Cleanup(func() { require.NoError(t, svc.Close()) })
					svc.SetPersistenceEnabled(true)
					svc.SetFulltextIndexPath(path)
					require.NoError(t, svc.BuildIndexes(context.Background()))
					require.True(t, svc.IsReady())
					if phase.reuse {
						require.Zero(t, index.indexCalls, "matching BM25 snapshot must be reused")
					} else {
						require.Positive(t, index.indexCalls, "changed analysis must rebuild BM25 from storage")
					}
					for query, want := range map[string]bool{
						"вагонами": phase.language == "russian",
						"елками":   phase.language == "russian",
						"runs":     phase.language == "english",
					} {
						response, err := svc.Search(context.Background(), query, nil, nil)
						require.NoError(t, err)
						if want {
							require.Len(t, response.Results, 1, query)
							require.Equal(t, "wagon", response.Results[0].ID)
						} else {
							require.Empty(t, response.Results, query)
						}
					}
					text, ok := svc.fulltextIndex.GetDocument("wagon")
					require.True(t, ok)
					require.Contains(t, text, "вагон Ёлка running")
					settings, err := loadSearchBuildSettings(searchBuildSettingsPath(path, "", ""))
					require.NoError(t, err)
					require.NotNil(t, settings)
					require.Equal(t, svc.composeBM25BuildSettings(), settings.BM25)
				})
			}
		})
	}
}
