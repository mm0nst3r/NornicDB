package search

import (
	"path/filepath"
	"strings"
	"testing"
)

type stemmingIndexFactory struct {
	name string
	new  func(string) (bm25Index, error)
}

var stemmingIndexFactories = []stemmingIndexFactory{
	{"v1", func(language string) (bm25Index, error) { return NewFulltextIndexWithStemmer(language) }},
	{"v2", func(language string) (bm25Index, error) { return NewFulltextIndexV2WithStemmer(language) }},
}

func newStemmingTestIndex(t *testing.T, factory stemmingIndexFactory, language string) bm25Index {
	t.Helper()
	idx, err := factory.new(language)
	if err != nil {
		t.Fatal(err)
	}
	return idx
}

func requireStemmedHit(t *testing.T, idx bm25Index, query, id string) {
	t.Helper()
	hits := idx.Search(query, 10)
	if len(hits) != 1 || hits[0].ID != id || hits[0].Score <= 0 {
		t.Fatalf("Search(%q)=%v, want one positive-scoring hit %q", query, hits, id)
	}
}

func TestBM25StemmingIndexQueryAndOriginalText(t *testing.T) {
	for _, factory := range stemmingIndexFactories {
		t.Run(factory.name, func(t *testing.T) {
			plain := newStemmingTestIndex(t, factory, "none")
			stemmed := newStemmingTestIndex(t, factory, "ru")
			for _, idx := range []bm25Index{plain, stemmed} {
				idx.Index("wagon", "вагон")
				idx.IndexBatch([]FulltextBatchEntry{{ID: "tree", Text: "Ёлка"}})
			}
			// The reverse direction can match through existing prefix expansion.
			// An inflected query against singular text proves actual stemming.
			if hits := plain.Search("вагоны", 10); len(hits) != 0 {
				t.Fatalf("neutral analysis unexpectedly matched: %v", hits)
			}
			requireStemmedHit(t, stemmed, "вагоны", "wagon")
			requireStemmedHit(t, stemmed, "вагона", "wagon")
			requireStemmedHit(t, stemmed, "елки", "tree")
			if text, ok := stemmed.GetDocument("tree"); !ok || text != "Ёлка" {
				t.Fatalf("original text changed: %q, exists=%t", text, ok)
			}
			if hits := stemmed.PhraseSearch("вагоны", 10); len(hits) != 0 {
				t.Fatalf("literal phrase search must not stem: %v", hits)
			}
			if hits := stemmed.PhraseSearch("вагон", 10); len(hits) != 1 {
				t.Fatalf("literal phrase disappeared: %v", hits)
			}
		})
	}
}

func TestBM25StemmingMutations(t *testing.T) {
	for _, factory := range stemmingIndexFactories {
		t.Run(factory.name, func(t *testing.T) {
			idx := newStemmingTestIndex(t, factory, "ru")
			idx.IndexBatch([]FulltextBatchEntry{
				{ID: "a", Text: "вагон вагоны вагона"},
				{ID: "b", Text: "самолет"},
			})
			requireStemmedHit(t, idx, "вагонами", "a")
			switch concrete := idx.(type) {
			case *FulltextIndex:
				if concrete.docLengths["a"] != 3 || concrete.invertedIndex["вагон"]["a"] != 3 {
					t.Fatal("V1 did not combine stem frequencies while preserving length")
				}
			case *FulltextIndexV2:
				if concrete.docLengths[concrete.docIDToNum["a"]] != 3 || concrete.termIndex["вагон"].Postings[0].TF != 3 {
					t.Fatal("V2 did not combine stem frequencies while preserving length")
				}
			}
			idx.Index("a", "корабль")
			if hits := idx.Search("вагоны", 10); len(hits) != 0 {
				t.Fatalf("update left old postings: %v", hits)
			}
			requireStemmedHit(t, idx, "корабли", "a")
			idx.IndexBatch([]FulltextBatchEntry{{ID: "a", Text: "вагон"}, {ID: "a", Text: "Ёлка"}})
			if hits := idx.Search("вагоны", 10); len(hits) != 0 {
				t.Fatalf("batch update left old postings: %v", hits)
			}
			requireStemmedHit(t, idx, "елки", "a")
			idx.Remove("a")
			idx.Remove("missing")
			if hits := idx.Search("елки", 10); len(hits) != 0 || idx.Count() != 1 {
				t.Fatalf("remove left postings/count: %v, %d", hits, idx.Count())
			}
			idx.Clear()
			idx.Index("new", "вагон")
			requireStemmedHit(t, idx, "вагоны", "new")
		})
	}
}

func TestBM25StemmingSnapshotRoundTripAndMismatch(t *testing.T) {
	for _, factory := range stemmingIndexFactories {
		for _, noCopy := range []bool{false, true} {
			name := factory.name + "/save"
			if noCopy {
				name += "-no-copy"
			}
			t.Run(name, func(t *testing.T) {
				for _, language := range []string{"none", "russian", "english"} {
					t.Run(language, func(t *testing.T) {
						source := newStemmingTestIndex(t, factory, language)
						source.Index("a", "вагон running Ёлка")
						path := filepath.Join(t.TempDir(), "bm25")
						var err error
						if noCopy {
							err = source.SaveNoCopy(path)
						} else {
							err = source.Save(path)
						}
						if err != nil {
							t.Fatal(err)
						}
						if source.IsDirty() {
							t.Fatal("successful save must clear dirty state")
						}
						for _, targetLanguage := range []string{"none", "russian", "english"} {
							target := newStemmingTestIndex(t, factory, targetLanguage)
							target.Index("stale", "stale")
							_ = target.Search("stale", 10) // Prime V2's query-plan cache.
							if err := target.Load(path); err != nil {
								t.Fatal(err)
							}
							if language != targetLanguage {
								if target.Count() != 0 || len(target.Search("stale", 10)) != 0 {
									t.Fatalf("accepted %s snapshot into %s", language, targetLanguage)
								}
								// Empty index is the existing caller-rebuild signal.
								target.Index("rebuilt", "вагон")
								if targetLanguage == "russian" {
									requireStemmedHit(t, target, "вагоны", "rebuilt")
								}
								continue
							}
							if target.Count() != 1 || target.IsDirty() {
								t.Fatal("matching snapshot did not load cleanly")
							}
							if text, _ := target.GetDocument("a"); text != "вагон running Ёлка" {
								t.Fatalf("original text did not round-trip: %q", text)
							}
							switch language {
							case "russian":
								requireStemmedHit(t, target, "вагоны", "a")
							case "english":
								requireStemmedHit(t, target, "runs", "a")
							}
						}
					})
				}
			})
		}
	}
}

func TestBM25StemmingV1ToV2Migration(t *testing.T) {
	for _, language := range []string{"none", "russian"} {
		for _, targetLanguage := range []string{"none", "russian"} {
			t.Run(language+"-to-"+targetLanguage, func(t *testing.T) {
				source, err := NewFulltextIndexWithStemmer(language)
				if err != nil {
					t.Fatal(err)
				}
				source.Index("a", "вагон")
				path := filepath.Join(t.TempDir(), "bm25")
				if err := source.Save(path); err != nil {
					t.Fatal(err)
				}
				target, err := NewFulltextIndexV2WithStemmer(targetLanguage)
				if err != nil {
					t.Fatal(err)
				}
				if err := target.Load(path); err != nil {
					t.Fatal(err)
				}
				want := 0
				if language == targetLanguage {
					want = 1
				}
				if target.Count() != want {
					t.Fatalf("migration count=%d, want %d", target.Count(), want)
				}
				if want == 1 && language == "russian" {
					requireStemmedHit(t, target, "вагоны", "a")
				}
			})
		}
	}
}

func TestBM25StemmingLegacySnapshotRebuilds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy")
	legacy := bm25V1Snapshot{
		Version: "1.0.0", Documents: map[string]string{"a": "вагон"},
		InvertedIndex: map[string]map[string]int{"вагон": {"a": 1}},
		DocLengths:    map[string]int{"a": 1}, AvgDocLength: 1, DocCount: 1,
	}
	if err := writeMsgpackSnapshot(path, &legacy); err != nil {
		t.Fatal(err)
	}
	for _, language := range []string{"none", "russian"} {
		idx, err := NewFulltextIndexV2WithStemmer(language)
		if err != nil {
			t.Fatal(err)
		}
		if err := idx.Load(path); err != nil {
			t.Fatal(err)
		}
		if idx.Count() != 0 {
			t.Fatalf("legacy analyzer must rebuild for %s: count=%d", language, idx.Count())
		}
	}
}

func TestBM25StemmingMissingAnalyzerRebuilds(t *testing.T) {
	for _, factory := range stemmingIndexFactories {
		t.Run(factory.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "missing-analyzer")
			var snapshot any = fulltextIndexSnapshot{
				Version: fulltextIndexFormatVersion, Documents: map[string]string{"a": "text"}, DocCount: 1,
			}
			if factory.name == "v2" {
				snapshot = bm25V2Snapshot{Version: bm25V2FormatVersion, Documents: map[string]string{"a": "text"}, DocCount: 1}
			}
			if err := writeMsgpackSnapshot(path, snapshot); err != nil {
				t.Fatal(err)
			}
			idx := newStemmingTestIndex(t, factory, "none")
			idx.Index("stale", "stale")
			if err := idx.Load(path); err != nil {
				t.Fatal(err)
			}
			if idx.Count() != 0 {
				t.Fatal("accepted missing analyzer identity")
			}
		})
	}
}

func TestBM25StemmingEnvironmentAndBuildSettingsAreImmutable(t *testing.T) {
	for _, engine := range []string{BM25EngineV1, BM25EngineV2} {
		t.Run(engine, func(t *testing.T) {
			t.Setenv(EnvBM25Stemmer, "ru")
			idx, _ := newBM25Index(engine)
			svc := &Service{fulltextIndex: idx, bm25Engine: engine}
			before := svc.composeBM25BuildSettings()
			if !strings.Contains(before, ":russian;") {
				t.Fatalf("missing analyzer in fingerprint: %q", before)
			}
			t.Setenv(EnvBM25Stemmer, "none")
			idx.Index("a", "вагон")
			requireStemmedHit(t, idx, "вагоны", "a")
			if after := svc.composeBM25BuildSettings(); after != before {
				t.Fatalf("live environment changed identity: %q != %q", after, before)
			}
			plain, _ := newBM25Index(engine)
			plain.Index("a", "вагон")
			if len(plain.Search("вагоны", 10)) != 0 {
				t.Fatal("new neutral index inherited old analyzer")
			}
			plainService := &Service{fulltextIndex: plain, bm25Engine: engine}
			plainSettings := plainService.composeBM25BuildSettings()
			if bm25SettingsEquivalent(before, plainSettings, svc.currentBM25FormatVersion()) {
				t.Fatal("settings equivalence ignored analyzer mismatch")
			}
		})
	}
}
