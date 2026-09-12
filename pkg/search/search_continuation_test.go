package search

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/orneryd/nornicdb/pkg/knowledgepolicy"
	"github.com/orneryd/nornicdb/pkg/storage"
)

func continuationTestService(t *testing.T) (*Service, storage.Engine) {
	t.Helper()
	base := storage.NewMemoryEngine()
	engine := storage.NewNamespacedEngine(base, "continuation-test")
	svc := NewServiceWithDimensions(engine, 3)
	t.Cleanup(func() { _ = svc.Close(); _ = base.Close() })
	return svc, engine
}

func continuationTestCreate(t *testing.T, svc *Service, engine storage.Engine, id, parent, collection string, vector []float32) {
	t.Helper()
	node := &storage.Node{ID: storage.NodeID(id), Labels: []string{"AssetFrame"}, Properties: map[string]any{
		"parent": parent, "collection": collection, "content": "beach sunset photograph", "tags": []string{"blue", "sea"},
	}}
	if len(vector) > 0 {
		node.ChunkEmbeddings = [][]float32{vector}
	}
	finish := svc.BeginSearchContinuationMutation()
	defer finish()
	if _, err := engine.CreateNode(node); err != nil {
		t.Fatal(err)
	}
	if err := svc.IndexNode(node); err != nil {
		t.Fatal(err)
	}
}

func TestSearchPageIDGroupingFiltersAndMutation(t *testing.T) {
	ctx := context.Background()
	svc, engine := continuationTestService(t)
	for _, r := range [][3]string{{"a2", "A", "summer"}, {"a1", "A", "summer"}, {"b1", "B", "summer"}, {"c1", "C", "summer"}, {"excluded", "D", "winter"}} {
		continuationTestCreate(t, svc, engine, r[0], r[1], r[2], nil)
	}
	opts := DefaultSearchOptions()
	opts.Types = []string{"assetframe"}
	opts.Filters = map[string][]string{"collection": {"summer"}, "tags": {"sea"}}
	paging := &SearchPageOptions{Mode: SearchPageID, GroupBy: "parent", PageSize: 1, Scope: "test-reader:v1"}
	page, err := svc.SearchPage(ctx, "", nil, opts, paging)
	if err != nil {
		t.Fatal(err)
	}
	if page.Total != 3 || page.Results[0].ID != "a1" || page.Results[0].GroupKey != "A" || page.Results[0].Phase != "catalog" {
		t.Fatalf("bad page: %+v", page)
	}
	paging.Cursor = page.NextCursor
	replay, err := svc.SearchPage(ctx, "", nil, opts, paging)
	if err != nil {
		t.Fatal(err)
	}
	if replay.Results[0].GroupKey != "B" {
		t.Fatal(replay)
	}
	wrong := *paging
	wrong.Scope = "other-user"
	if _, err := svc.SearchPage(ctx, "", nil, opts, &wrong); !errors.Is(err, ErrSearchCursorMismatch) {
		t.Fatal(err)
	}
	paging.PageSize = 2
	last, err := svc.SearchPage(ctx, "", nil, opts, paging)
	if err != nil {
		t.Fatal(err)
	}
	if len(last.Results) != 2 || !last.CollectionExhausted {
		t.Fatal(last)
	}
	node, err := engine.GetNode("c1")
	if err != nil {
		t.Fatal(err)
	}
	finish := svc.BeginSearchContinuationMutation()
	node.Properties["collection"] = "winter"
	if err := engine.UpdateNode(node); err != nil {
		t.Fatal(err)
	}
	if err := svc.IndexNode(node); err != nil {
		t.Fatal(err)
	}
	finish()
	if _, err := svc.SearchPage(ctx, "", nil, opts, paging); !errors.Is(err, ErrSearchCursorInvalidated) {
		t.Fatal(err)
	}
}

func TestSearchPageVectorAndHybridContinuePastUnselectedAssets(t *testing.T) {
	// Run these against the real retrieval implementation, not a cursor stub.
	for _, query := range []string{"", "beach"} {
		t.Run(fmt.Sprintf("query=%q", query), func(t *testing.T) {
			ctx := context.Background()
			svc, engine := continuationTestService(t)
			continuationTestCreate(t, svc, engine, "v1", "A", "summer", []float32{1, 0, 0})
			continuationTestCreate(t, svc, engine, "v2", "B", "summer", []float32{0.8, 0.2, 0})
			continuationTestCreate(t, svc, engine, "v3", "C", "summer", []float32{0, 1, 0})
			continuationTestCreate(t, svc, engine, "no-vector", "D", "summer", nil)
			continuationTestCreate(t, svc, engine, "not-eligible", "E", "winter", []float32{1, 0, 0})
			opts := DefaultSearchOptions()
			opts.Limit = 1
			opts.CandidateTarget = 1
			opts.MaxCandidateLimit = 10
			opts.Filters = map[string][]string{"collection": {"summer"}}
			threshold := -1.0
			opts.MinSimilarity = &threshold
			opts.MinRRFScore = 0
			paging := &SearchPageOptions{Mode: SearchPageRankedThenID, PageSize: 1, Scope: "reader"}
			page, err := svc.SearchPage(ctx, query, []float32{1, 0, 0}, opts, paging)
			if err != nil {
				t.Fatal(err)
			}
			if page.Total != 4 || page.RankedCount != 1 || page.Results[0].Phase != "ranked" {
				t.Fatalf("unexpected prefix: %+v", page)
			}
			seen := map[string]bool{}
			for {
				for _, hit := range page.Results {
					if seen[hit.ID] {
						t.Fatal("duplicate", hit.ID)
					}
					seen[hit.ID] = true
				}
				if page.NextCursor == "" {
					break
				}
				paging.Cursor = page.NextCursor
				page, err = svc.SearchPage(ctx, query, []float32{1, 0, 0}, opts, paging)
				if err != nil {
					t.Fatal(err)
				}
			}
			if len(seen) != 4 || !seen["no-vector"] || seen["not-eligible"] || !page.CollectionExhausted {
				t.Fatal(seen, page)
			}
			paging.Mode = SearchPageRanked
			paging.Cursor = ""
			paging.PageSize = 50
			bounded, err := svc.SearchPage(ctx, query, []float32{1, 0, 0}, opts, paging)
			if err != nil {
				t.Fatal(err)
			}
			if bounded.CollectionExhausted || bounded.EligibleCount != nil || bounded.Completion != "candidate_pool_exhausted" {
				t.Fatal(bounded)
			}
		})
	}
}

func TestSearchPageRequestCanonicalizationAndOwnership(t *testing.T) {
	svc, _ := continuationTestService(t)
	opts := DefaultSearchOptions()
	opts.Types = []string{"B", "a", "A"}
	opts.Filters = map[string][]string{"x": {"b", "a", "b"}, "ignored": {}}
	vector := []float32{1, 0, 0}
	paging := &SearchPageOptions{Scope: "reader"}
	beforeTypes := append([]string(nil), opts.Types...)
	beforeFilter := append([]string(nil), opts.Filters["x"]...)
	a, err := svc.resolveSearchPageRequest("query", vector, opts, paging)
	if err != nil {
		t.Fatal(err)
	}
	other := *opts
	other.Types = []string{"a", "b"}
	other.Filters = map[string][]string{"x": {"a", "b"}}
	b, err := svc.resolveSearchPageRequest("query", vector, &other, paging)
	if err != nil {
		t.Fatal(err)
	}
	if a.fingerprint != b.fingerprint {
		t.Fatal("equivalent sets not canonical")
	}
	if opts.MinSimilarity != nil || !reflect.DeepEqual(beforeTypes, opts.Types) || !reflect.DeepEqual(beforeFilter, opts.Filters["x"]) {
		t.Fatal("caller options modified")
	}
	vector[0] = 7
	opts.Filters["x"][0] = "changed"
	if a.embedding[0] != 1 || a.options.Filters["x"][1] != "b" {
		t.Fatal("request aliases caller")
	}
	// Delimiter-containing values MUST NOT collide with multiple values.
	left := DefaultSearchOptions()
	left.Filters = map[string][]string{"x": {"a,b"}}
	right := DefaultSearchOptions()
	right.Filters = map[string][]string{"x": {"a", "b"}}
	l, _ := svc.resolveSearchPageRequest("q", nil, left, paging)
	r, _ := svc.resolveSearchPageRequest("q", nil, right, paging)
	if l.fingerprint == r.fingerprint {
		t.Fatal("filter key collision")
	}
	bad := DefaultSearchOptions()
	bad.MMREnabled = true
	if _, err := svc.SearchPage(context.Background(), "", nil, bad, paging); !errors.Is(err, ErrSearchPageRequest) {
		t.Fatal(err)
	}
	if _, err := svc.SearchPage(context.Background(), "", []float32{float32(math.NaN())}, nil, paging); !errors.Is(err, ErrSearchPageRequest) {
		t.Fatal(err)
	}
	if _, err := svc.SearchPage(context.Background(), "", nil, nil, nil); !errors.Is(err, ErrSearchPageRequest) {
		t.Fatal(err)
	}
}

func TestSearchPageMissingGroupsAndCapacityFailClosed(t *testing.T) {
	svc, engine := continuationTestService(t)
	continuationTestCreate(t, svc, engine, "a", "", "summer", nil)
	pageOpts := &SearchPageOptions{Mode: SearchPageID, GroupBy: "parent", Scope: "reader"}
	if _, err := svc.SearchPage(context.Background(), "", nil, nil, pageOpts); !errors.Is(err, ErrSearchPageRequest) {
		t.Fatal(err)
	}
	continuationTestCreate(t, svc, engine, "b", "B", "summer", nil)
	if err := svc.ConfigureSearchContinuation(SearchContinuationConfig{MaxResults: 1}); err != nil {
		t.Fatal(err)
	}
	pageOpts.GroupBy = ""
	if page, err := svc.SearchPage(context.Background(), "", nil, nil, pageOpts); !errors.Is(err, ErrSearchContinuationLimit) || page != nil {
		t.Fatal(page, err)
	}
}

func TestSearchPageLifecycleHooks(t *testing.T) {
	for _, mutation := range []string{"index", "remove", "build", "clear", "property", "decay", "close"} {
		t.Run(mutation, func(t *testing.T) {
			svc, engine := continuationTestService(t)
			continuationTestCreate(t, svc, engine, "a", "A", "summer", nil)
			continuationTestCreate(t, svc, engine, "b", "B", "summer", nil)
			paging := &SearchPageOptions{Mode: SearchPageID, PageSize: 1, Scope: "reader"}
			page, err := svc.SearchPage(context.Background(), "", nil, nil, paging)
			if err != nil {
				t.Fatal(err)
			}
			paging.Cursor = page.NextCursor
			switch mutation {
			case "index":
				node, _ := engine.GetNode("a")
				if err := svc.IndexNode(node); err != nil {
					t.Fatal(err)
				}
			case "remove":
				if err := svc.RemoveNode("a"); err != nil {
					t.Fatal(err)
				}
			case "build":
				if err := svc.BuildIndexes(context.Background()); err != nil {
					t.Fatal(err)
				}
			case "clear":
				svc.ClearVectorIndex()
			case "property":
				svc.RemovePropertyVectorIndex("embedding")
			case "decay":
				svc.SetNodeDecayFilter(func(string) bool { return false })
			case "close":
				if err := svc.Close(); err != nil {
					t.Fatal(err)
				}
			}
			_, err = svc.SearchPage(context.Background(), "", nil, nil, paging)
			if mutation == "close" {
				if !errors.Is(err, ErrSearchContinuationClosed) {
					t.Fatal(err)
				}
			} else if !errors.Is(err, ErrSearchCursorInvalidated) {
				t.Fatal(err)
			}
		})
	}
}

func TestSearchPageNoOpConfigurationDoesNotInvalidate(t *testing.T) {
	svc, engine := continuationTestService(t)
	continuationTestCreate(t, svc, engine, "a", "A", "summer", nil)
	continuationTestCreate(t, svc, engine, "b", "B", "summer", nil)
	paging := &SearchPageOptions{Mode: SearchPageID, PageSize: 1, Scope: "reader"}
	page, err := svc.SearchPage(context.Background(), "", nil, nil, paging)
	if err != nil {
		t.Fatal(err)
	}
	paging.Cursor = page.NextCursor
	svc.SetReranker(nil)
	if svc.SetIndexFlags(true, true) {
		t.Fatal("default master flags unexpectedly changed")
	}
	if _, err := svc.SearchPage(context.Background(), "", nil, nil, paging); err != nil {
		t.Fatal("no-op setter invalidated cursor", err)
	}
	if !svc.SetIndexFlags(false, true) {
		t.Fatal("master flag change not reported")
	}
	if _, err := svc.SearchPage(context.Background(), "", nil, nil, paging); !errors.Is(err, ErrSearchCursorInvalidated) {
		t.Fatal(err)
	}
}

func TestSearchPageCompletedStorageMutationInvalidates(t *testing.T) {
	for _, backend := range []string{"memory", "badger", "async", "wal", "traced"} {
		t.Run(backend, func(t *testing.T) {
			for _, operation := range []string{"create", "update", "delete", "bulk-create", "bulk-delete", "delete-prefix"} {
				t.Run(operation, func(t *testing.T) {
					var base storage.Engine = storage.NewMemoryEngine()
					if backend == "badger" {
						_ = base.Close()
						var err error
						base, err = storage.NewBadgerEngine(t.TempDir())
						if err != nil {
							t.Fatal(err)
						}
					}
					switch backend {
					case "async":
						base = storage.NewAsyncEngine(base, &storage.AsyncEngineConfig{FlushInterval: time.Hour})
					case "wal":
						wal, err := storage.NewWAL(t.TempDir(), nil)
						if err != nil {
							t.Fatal(err)
						}
						base = storage.NewWALEngine(base, wal)
					case "traced":
						base = storage.NewTracedEngine(base)
					}
					engine := storage.NewNamespacedEngine(base, "continuation-test")
					svc := NewServiceWithDimensions(engine, 3)
					t.Cleanup(func() { _ = svc.Close(); _ = base.Close() })
					continuationTestCreate(t, svc, engine, "a", "A", "summer", nil)
					continuationTestCreate(t, svc, engine, "b", "B", "summer", nil)
					paging := &SearchPageOptions{Mode: SearchPageID, PageSize: 1, Scope: "reader"}
					page, err := svc.SearchPage(context.Background(), "", nil, nil, paging)
					if err != nil {
						t.Fatal(err)
					}
					paging.Cursor = page.NextCursor
					// Built-in storage writes may become visible before deferred indexing.
					// A completed write must invalidate the population without requiring
					// an application to know and bracket the search service internals.
					switch operation {
					case "create":
						_, err = engine.CreateNode(&storage.Node{ID: "c", Labels: []string{"AssetFrame"}})
					case "update":
						var node *storage.Node
						node, err = engine.GetNode("b")
						if err == nil {
							node.Properties["collection"] = "winter"
							err = engine.UpdateNode(node)
						}
					case "delete":
						err = engine.DeleteNode("b")
					case "bulk-create":
						err = engine.BulkCreateNodes([]*storage.Node{{ID: "c", Labels: []string{"AssetFrame"}}})
					case "bulk-delete":
						err = engine.BulkDeleteNodes([]storage.NodeID{"b"})
					case "delete-prefix":
						_, _, err = base.DeleteByPrefix("continuation-test:b")
					}
					if err != nil {
						t.Fatal(err)
					}
					_, err = svc.SearchPage(context.Background(), "", nil, nil, paging)
					if !errors.Is(err, ErrSearchCursorInvalidated) {
						t.Fatalf("completed %s: want invalidated cursor, got %v", operation, err)
					}
				})
			}
		})
	}
}

func TestSearchPageOtherDatabaseWritePreservesCursor(t *testing.T) {
	base := storage.NewMemoryEngine()
	t.Cleanup(func() { _ = base.Close() })
	engine := storage.NewNamespacedEngine(base, "first")
	other := storage.NewNamespacedEngine(base, "second")
	svc := NewServiceWithDimensions(engine, 3)
	t.Cleanup(func() { _ = svc.Close() })
	continuationTestCreate(t, svc, engine, "a", "A", "summer", nil)
	continuationTestCreate(t, svc, engine, "b", "B", "summer", nil)
	paging := &SearchPageOptions{Mode: SearchPageID, PageSize: 1, Scope: "reader"}
	page, err := svc.SearchPage(context.Background(), "", nil, nil, paging)
	if err != nil {
		t.Fatal(err)
	}
	paging.Cursor = page.NextCursor
	if _, err = other.CreateNode(&storage.Node{ID: "a"}); err != nil {
		t.Fatal(err)
	}
	if _, err = svc.SearchPage(context.Background(), "", nil, nil, paging); err != nil {
		t.Fatalf("other database invalidated cursor: %v", err)
	}
}

func TestSearchPageNestedNamespaceMutationInvalidates(t *testing.T) {
	base := storage.NewMemoryEngine()
	t.Cleanup(func() { _ = base.Close() })
	engine := storage.NewNamespacedEngine(storage.NewNamespacedEngine(base, "physical"), "view")
	svc := NewServiceWithDimensions(engine, 3)
	t.Cleanup(func() { _ = svc.Close() })
	continuationTestCreate(t, svc, engine, "a", "A", "summer", nil)
	continuationTestCreate(t, svc, engine, "b", "B", "summer", nil)
	paging := &SearchPageOptions{Mode: SearchPageID, PageSize: 1, Scope: "reader"}
	page, err := svc.SearchPage(context.Background(), "", nil, nil, paging)
	if err != nil {
		t.Fatal(err)
	}
	paging.Cursor = page.NextCursor
	if err = engine.DeleteNode("b"); err != nil {
		t.Fatal(err)
	}
	_, err = svc.SearchPage(context.Background(), "", nil, nil, paging)
	if !errors.Is(err, ErrSearchCursorInvalidated) {
		t.Fatalf("nested view missed mutation: %v", err)
	}
}

// Embedding Engine deliberately exposes only the public Engine contract, like
// an external implementation without built-in revision/streaming capabilities.
type continuationExternalEngine struct{ storage.Engine }

func TestSearchPageExternalEngineExplicitMutationContract(t *testing.T) {
	base := storage.NewMemoryEngine()
	t.Cleanup(func() { _ = base.Close() })
	engine := storage.NewNamespacedEngine(&continuationExternalEngine{Engine: base}, "external")
	svc := NewServiceWithDimensions(engine, 3)
	t.Cleanup(func() { _ = svc.Close() })
	continuationTestCreate(t, svc, engine, "a", "A", "summer", nil)
	continuationTestCreate(t, svc, engine, "b", "B", "summer", nil)
	paging := &SearchPageOptions{Mode: SearchPageID, PageSize: 1, Scope: "reader"}
	page, err := svc.SearchPage(context.Background(), "", nil, nil, paging)
	if err != nil {
		t.Fatal(err)
	}
	paging.Cursor = page.NextCursor
	finish := svc.BeginSearchContinuationMutation()
	err = engine.DeleteNode("b")
	finish()
	if err != nil {
		t.Fatal(err)
	}
	_, err = svc.SearchPage(context.Background(), "", nil, nil, paging)
	if !errors.Is(err, ErrSearchCursorInvalidated) {
		t.Fatalf("external bracket did not invalidate: %v", err)
	}
}

func TestSearchPageNativeEligibilityPolicyInvalidates(t *testing.T) {
	base := storage.NewMemoryEngine()
	t.Cleanup(func() { _ = base.Close() })
	base.SetDecayEnabled(true)
	engine := storage.NewNamespacedEngine(base, "policy-pages")
	schema := base.GetSchemaForNamespace("policy-pages")
	if err := schema.CreateDecayProfileBundle(knowledgepolicy.DecayProfileBundle{Name: "retention", Scope: knowledgepolicy.ScopeNode, Function: knowledgepolicy.DecayFunctionExponential, HalfLifeSeconds: 10000000, VisibilityThreshold: .1, ScoreFrom: knowledgepolicy.ScoreFromCreated, Enabled: true, DecayEnabled: true}); err != nil {
		t.Fatal(err)
	}
	if err := schema.CreateDecayProfileBinding(knowledgepolicy.DecayProfileBinding{Name: "documents", ProfileRef: "retention", TargetLabels: []string{"Document"}}); err != nil {
		t.Fatal(err)
	}
	for _, id := range []storage.NodeID{"a", "b"} {
		_, err := engine.CreateNode(&storage.Node{ID: id, Labels: []string{"Document"}, CreatedAt: time.Now().Add(-72 * time.Hour)})
		if err != nil {
			t.Fatal(err)
		}
	}
	svc := NewServiceWithDimensions(engine, 3)
	t.Cleanup(func() { _ = svc.Close() })
	opts := &SearchPageOptions{Mode: SearchPageID, Scope: "reader", PageSize: 1}
	page, err := svc.SearchPage(context.Background(), "", nil, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if page.Total != 2 {
		t.Fatalf("initial population: %+v", page)
	}
	opts.Cursor = page.NextCursor
	if err := schema.AlterDecayProfile("retention", map[string]interface{}{"halfLifeSeconds": 1}); err != nil {
		t.Fatal(err)
	}
	_, err = svc.SearchPage(context.Background(), "", nil, nil, opts)
	if !errors.Is(err, ErrSearchCursorInvalidated) {
		t.Fatalf("policy changed eligibility without invalidating: %v", err)
	}
}

func TestSearchPageReportsCandidateStopWithoutClaimingCollectionExhaustion(t *testing.T) {
	svc, engine := continuationTestService(t)
	for i := 0; i < 12; i++ {
		collection := "winter"
		if i == 11 {
			collection = "summer"
		}
		continuationTestCreate(t, svc, engine, fmt.Sprintf("%02d", i), "", ""+collection, nil)
	}
	opts := DefaultSearchOptions()
	opts.Limit = 2
	opts.CandidateTarget = 2
	opts.InitialOverfetchRatio = 1
	opts.MaxCandidateLimit = 3
	opts.Filters = map[string][]string{"collection": {"not-present"}}
	page, err := svc.SearchPage(context.Background(), "beach", nil, opts, &SearchPageOptions{Mode: SearchPageRanked, Scope: "reader"})
	if err != nil {
		t.Fatal(err)
	}
	if page.BM25StopReason != "candidate_limit" || page.BM25CandidateLimit != 3 || page.CollectionExhausted || page.EligibleCount != nil {
		t.Fatalf("wrong budget report: %+v", page)
	}
	page, err = svc.SearchPage(context.Background(), "absentterm", nil, opts, &SearchPageOptions{Mode: SearchPageRanked, Scope: "reader"})
	if err != nil {
		t.Fatal(err)
	}
	if page.BM25StopReason != "short_response" || page.CollectionExhausted {
		t.Fatalf("short response claimed budget/corpus exhaustion: %+v", page)
	}
}
