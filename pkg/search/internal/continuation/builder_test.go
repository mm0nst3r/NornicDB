package continuation

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestBuilderGroupingStableAcrossScanOrders(t *testing.T) {
	ranked := []Hit{{ID: "p3", Score: 0.9}, {ID: "p2", Score: 0.9}, {ID: "p1", Score: 0.8}, {ID: "p4", Score: 0.9}}
	nodes := [][2]string{{"p1", "A"}, {"p2", "A"}, {"p3", "B"}, {"p4", "C"}, {"t2", "D"}, {"t1", "D"}, {"x", "B"}, {"no-vector", "E"}}
	want := []Hit{{ID: "p2", GroupKey: "A", Phase: RankedPhase, Score: 0.9}, {ID: "p3", GroupKey: "B", Phase: RankedPhase, Score: 0.9}, {ID: "p4", GroupKey: "C", Phase: RankedPhase, Score: 0.9}, {ID: "t1", GroupKey: "D", Phase: CatalogPhase}, {ID: "no-vector", GroupKey: "E", Phase: CatalogPhase}}
	rng := rand.New(rand.NewSource(346))
	for iteration := 0; iteration < 100; iteration++ {
		rng.Shuffle(len(nodes), func(i, j int) { nodes[i], nodes[j] = nodes[j], nodes[i] })
		rng.Shuffle(len(ranked), func(i, j int) { ranked[i], ranked[j] = ranked[j], ranked[i] })
		b, err := NewBuilder(RankedThenID, true, ranked, Config{})
		if err != nil {
			t.Fatal(err)
		}
		for _, n := range nodes {
			if err := b.Add(n[0], n[1]); err != nil {
				t.Fatal(err)
			}
		}
		p, err := b.Finish(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(p.Hits, want) || p.RankedCount != 3 {
			t.Fatalf("iteration %d: %+v", iteration, p)
		}
	}
}

func TestBuilderModesDuplicatesAndErrors(t *testing.T) {
	ctx := context.Background()
	b, err := NewBuilder(RankedOnly, false, []Hit{{ID: "a", Score: 2}, {ID: "a", Score: 1}, {ID: "a", Score: 2, RRFScore: 1}}, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if err = b.Add("a", ""); err != nil {
		t.Fatal(err)
	}
	if err = b.Add("not-selected", ""); err != nil {
		t.Fatal(err)
	}
	p, err := b.Finish(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Hits) != 1 || p.Hits[0].RRFScore != 1 {
		t.Fatalf("bad duplicate resolution: %+v", p)
	}
	if err = b.Add("x", ""); !errors.Is(err, ErrTicketClosed) {
		t.Fatal(err)
	}
	if _, err = b.Finish(ctx); !errors.Is(err, ErrTicketClosed) {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name    string
		mode    Mode
		group   bool
		ranked  []Hit
		config  Config
		id, key string
		want    error
	}{
		{"mode", "bad", false, nil, Config{}, "", "", ErrInvalidRequest},
		{"id_with_rank", IDOnly, false, []Hit{{ID: "a"}}, Config{}, "", "", ErrInvalidRequest},
		{"negative_config", IDOnly, false, nil, Config{MaxResults: -1}, "", "", ErrInvalidRequest},
		{"rank_cap", RankedOnly, false, []Hit{{ID: "a"}, {ID: "b"}}, Config{MaxCandidates: 1}, "", "", ErrCapacity},
		{"rank_bytes", RankedOnly, false, []Hit{{ID: "a"}}, Config{MaxBuildBytes: 1}, "", "", ErrCapacity},
		{"rank_empty_id", RankedOnly, false, []Hit{{}}, Config{}, "", "", ErrInvalidRequest},
		{"rank_utf8", RankedOnly, false, []Hit{{ID: "\xff"}}, Config{}, "", "", ErrInvalidRequest},
		{"rank_nan", RankedOnly, false, []Hit{{ID: "a", Score: math.NaN()}}, Config{}, "", "", ErrInvalidRequest},
		{"missing_group", IDOnly, true, nil, Config{}, "a", "", ErrInvalidRequest},
		{"unexpected_group", IDOnly, false, nil, Config{}, "a", "parent", ErrInvalidRequest},
		{"bad_utf8", IDOnly, false, nil, Config{}, "\xff", "", ErrInvalidRequest},
		{"bad_group_utf8", IDOnly, true, nil, Config{}, "a", "\xff", ErrInvalidRequest},
		{"empty_id", IDOnly, false, nil, Config{}, "", "", ErrInvalidRequest},
		{"member_bytes", IDOnly, false, nil, Config{MaxBuildBytes: 1}, "a", "", ErrCapacity},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, err := NewBuilder(tc.mode, tc.group, tc.ranked, tc.config)
			if err == nil {
				err = b.Add(tc.id, tc.key)
			}
			if !errors.Is(err, tc.want) {
				t.Fatalf("got %v, want %v", err, tc.want)
			}
		})
	}
	for _, config := range []Config{{MaxResults: 1}, {MaxScannedNodes: 1}} {
		b, _ := NewBuilder(IDOnly, false, nil, config)
		if err := b.Add("a", ""); err != nil {
			t.Fatal(err)
		}
		if err := b.Add("b", ""); !errors.Is(err, ErrCapacity) {
			t.Fatal(err)
		}
	}
	b, _ = NewBuilder(IDOnly, false, nil, Config{})
	_ = b.Add("a", "")
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := b.Finish(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	b, _ = NewBuilder(IDOnly, false, nil, Config{})
	if _, err := b.Finish(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestRepresentativeDiagnosticTies(t *testing.T) {
	base := Hit{ID: "a", Score: 1, Similarity: 1, RRFScore: 1, VectorRank: 2, BM25Rank: 2}
	for _, change := range []func(*Hit){func(h *Hit) { h.Score = 2 }, func(h *Hit) { h.ID = "0" }, func(h *Hit) { h.RRFScore = 2 }, func(h *Hit) { h.Similarity = 2 }, func(h *Hit) { h.VectorRank = 1 }, func(h *Hit) { h.BM25Rank = 1 }} {
		next := base
		change(&next)
		if !betterRepresentative(next, base) || betterRepresentative(base, next) {
			t.Fatal(next)
		}
	}
}

func TestTwentyThousandMembersExactlyOnce(t *testing.T) {
	ctx := context.Background()
	var store Store
	ticket, err := store.Start(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Abort()
	ranked := []Hit{{ID: "frame-19999", Score: 1}, {ID: "frame-10000", Score: 0.9}}
	builder, err := NewBuilder(RankedThenID, true, ranked, ticket.Config())
	if err != nil {
		t.Fatal(err)
	}
	for i := 19999; i >= 0; i-- {
		if err := builder.Add(fmt.Sprintf("frame-%05d", i), fmt.Sprintf("asset-%05d", i)); err != nil {
			t.Fatal(err)
		}
	}
	pop, err := builder.Finish(ctx)
	if err != nil {
		t.Fatal(err)
	}
	page, err := ticket.Commit(ctx, [32]byte{}, pop, 50)
	if err != nil {
		t.Fatal(err)
	}
	seen := make(map[string]bool)
	pages := 0
	for {
		pages++
		for _, h := range page.Results {
			if seen[h.GroupKey] {
				t.Fatal("duplicate", h.GroupKey)
			}
			seen[h.GroupKey] = true
		}
		if page.NextCursor == "" {
			break
		}
		page, err = store.Continue(ctx, page.NextCursor, [32]byte{}, 50)
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(seen) != 20000 || pages != 400 || !page.CollectionExhausted {
		t.Fatalf("members=%d pages=%d %+v", len(seen), pages, page)
	}
}

func BenchmarkContinuationBuild20000(b *testing.B) {
	b.ReportAllocs()
	for iteration := 0; iteration < b.N; iteration++ {
		builder, _ := NewBuilder(IDOnly, false, nil, Config{})
		for i := 0; i < 20000; i++ {
			if err := builder.Add(fmt.Sprintf("asset-%05d", i), ""); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := builder.Finish(context.Background()); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkContinuationPage50(b *testing.B) {
	ctx := context.Background()
	var store Store
	pop := Population{Mode: IDOnly}
	for i := 0; i < 20000; i++ {
		pop.Hits = append(pop.Hits, Hit{ID: fmt.Sprintf("asset-%05d", i), Phase: CatalogPhase})
	}
	ticket, _ := store.Start(ctx)
	page, err := ticket.Commit(ctx, [32]byte{}, pop, 50)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.Continue(ctx, page.NextCursor, [32]byte{}, 50); err != nil {
			b.Fatal(err)
		}
	}
}
