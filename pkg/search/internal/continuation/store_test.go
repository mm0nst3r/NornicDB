package continuation

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func TestStorePagesReplayAndExhaustion(t *testing.T) {
	var s Store
	ctx := context.Background()
	ticket, err := s.Start(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Abort()
	pop := Population{Mode: RankedThenID, RankedCount: 2, Hits: []Hit{
		{ID: "b", Score: 0.9, Phase: RankedPhase},
		{ID: "a", Score: 0.8, Phase: RankedPhase},
		{ID: "c", Phase: CatalogPhase},
		{ID: "d", Phase: CatalogPhase},
		{ID: "e", Phase: CatalogPhase},
	}}
	fingerprint := [32]byte{1}
	first, err := ticket.Commit(ctx, fingerprint, pop, 2)
	if err != nil {
		t.Fatal(err)
	}
	if first.Exhausted || first.CollectionExhausted || first.NextCursor == "" {
		t.Fatalf("wrong first page: %+v", first)
	}
	second, err := s.Continue(ctx, first.NextCursor, fingerprint, 2)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := s.Continue(ctx, first.NextCursor, fingerprint, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(second, replay) {
		t.Fatal("same cursor must be replayable")
	}
	second.Results[0].ID = "tampered by caller"
	replay2, err := s.Continue(ctx, first.NextCursor, fingerprint, 2)
	if err != nil {
		t.Fatal(err)
	}
	if replay2.Results[0].ID != "c" {
		t.Fatal("returned slice aliases cached population")
	}
	last, err := s.Continue(ctx, replay.NextCursor, fingerprint, 10)
	if err != nil {
		t.Fatal(err)
	}
	if !last.Exhausted || !last.CollectionExhausted || last.NextCursor != "" || last.Completion != CollectionComplete {
		t.Fatalf("wrong completion: %+v", last)
	}
	if last.EligibleCount == nil || *last.EligibleCount != 5 {
		t.Fatalf("wrong eligible count: %+v", last)
	}
}

func TestStoreDoesNotClaimANNExhaustiveRecall(t *testing.T) {
	for _, count := range []int{0, 1, 3} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			var s Store
			ctx := context.Background()
			ticket, err := s.Start(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer ticket.Abort()
			pop := Population{Mode: RankedOnly, RankedCount: count}
			for i := 0; i < count; i++ {
				pop.Hits = append(pop.Hits, Hit{ID: fmt.Sprint(i), Score: 1, Phase: RankedPhase})
			}
			page, err := ticket.Commit(ctx, [32]byte{}, pop, 50)
			if err != nil {
				t.Fatal(err)
			}
			if !page.Exhausted || page.CollectionExhausted || page.EligibleCount != nil || page.Completion != CandidatePoolComplete {
				t.Fatalf("short ANN result falsely claims collection exhaustion: %+v", page)
			}
		})
	}
}

func TestStoreRejectsMutationDuringBuildAndAfterPage(t *testing.T) {
	var s Store
	ctx := context.Background()
	ticket, err := s.Start(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Abort()
	finish := s.BeginMutation()
	if _, err := s.Start(ctx); !errors.Is(err, ErrInvalidated) {
		t.Fatalf("start during write: %v", err)
	}
	finish()
	finish() // release is idempotent
	if _, err := ticket.Commit(ctx, [32]byte{}, Population{Mode: IDOnly}, 1); !errors.Is(err, ErrInvalidated) {
		t.Fatalf("commit across write: %v", err)
	}
	ticket.Abort()
	good, err := s.Start(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer good.Abort()
	page, err := good.Commit(ctx, [32]byte{}, Population{Mode: IDOnly, Hits: []Hit{{ID: "a", Phase: CatalogPhase}, {ID: "b", Phase: CatalogPhase}}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	end := s.BeginMutation()
	end()
	if _, err := s.Continue(ctx, page.NextCursor, [32]byte{}, 1); !errors.Is(err, ErrInvalidated) {
		t.Fatalf("cursor after write: %v", err)
	}
}
