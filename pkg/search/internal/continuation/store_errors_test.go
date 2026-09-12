package continuation

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"io"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
)

var testFingerprint = [32]byte{3, 4, 6}

func testPopulation() Population {
	return Population{Mode: IDOnly, Hits: []Hit{{ID: "a", Phase: CatalogPhase}, {ID: "b", Phase: CatalogPhase}, {ID: "c", Phase: CatalogPhase}}}
}
func putTestPage(t *testing.T, s *Store) *Page {
	t.Helper()
	ticket, err := s.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Abort()
	page, err := ticket.Commit(context.Background(), testFingerprint, testPopulation(), 1)
	if err != nil {
		t.Fatal(err)
	}
	return page
}

func TestCursorAuthenticationExpiryReleaseAndClose(t *testing.T) {
	ctx := context.Background()
	var s Store
	now := time.Unix(100, 0)
	s.clock = func() time.Time { return now }
	p := putTestPage(t, &s)
	if len(p.NextCursor) > 128 {
		t.Fatal("cursor grew with population")
	}
	bad := []string{"", strings.Repeat("x", 100000), "!" + p.NextCursor[1:], p.NextCursor + "="}
	raw, _ := base64.RawURLEncoding.DecodeString(p.NextCursor)
	for _, i := range []int{0, 1, 17, 25, 33, 64} {
		edited := append([]byte(nil), raw...)
		edited[i] ^= 1
		bad = append(bad, base64.RawURLEncoding.EncodeToString(edited))
	}
	for _, token := range bad {
		if _, err := s.Continue(ctx, token, testFingerprint, 1); !errors.Is(err, ErrInvalidCursor) {
			t.Fatalf("tampered cursor: %v", err)
		}
	}
	for _, size := range []int{-1, 0, 501} {
		if _, err := s.Continue(ctx, p.NextCursor, testFingerprint, size); !errors.Is(err, ErrInvalidRequest) {
			t.Fatal(err)
		}
	}
	if _, err := s.Continue(ctx, p.NextCursor, [32]byte{9}, 1); !errors.Is(err, ErrCursorMismatch) {
		t.Fatal(err)
	}
	if err := s.Release(p.NextCursor, [32]byte{9}); !errors.Is(err, ErrCursorMismatch) {
		t.Fatal(err)
	}
	if err := s.Release("garbage", testFingerprint); !errors.Is(err, ErrInvalidCursor) {
		t.Fatal(err)
	}
	var other Store
	putTestPage(t, &other)
	if _, err := other.Continue(ctx, p.NextCursor, testFingerprint, 1); !errors.Is(err, ErrInvalidCursor) {
		t.Fatal(err)
	}
	// Even a correctly signed out-of-range offset is rejected.
	s.mu.Lock()
	id, _, _, _ := s.decodeCursorLocked(p.NextCursor)
	outOfBounds := s.encodeCursorLocked(s.entries[id], math.MaxUint64)
	s.mu.Unlock()
	if _, err := s.Continue(ctx, outOfBounds, testFingerprint, 1); !errors.Is(err, ErrInvalidCursor) {
		t.Fatal(err)
	}
	now = now.Add(4 * time.Minute)
	p2, err := s.Continue(ctx, p.NextCursor, testFingerprint, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !p2.ExpiresAt.Equal(p.ExpiresAt) {
		t.Fatal("sliding TTL")
	}
	now = p.ExpiresAt
	if _, err := s.Continue(ctx, p.NextCursor, testFingerprint, 1); !errors.Is(err, ErrCursorExpired) {
		t.Fatal(err)
	}
	if s.bytes != 0 || len(s.entries) != 0 {
		t.Fatal("expiry did not reclaim descriptors")
	}
	if err := s.Release(p.NextCursor, testFingerprint); !errors.Is(err, ErrCursorExpired) {
		t.Fatal(err)
	}
	p = putTestPage(t, &s)
	if err := s.Release(p.NextCursor, testFingerprint); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Continue(ctx, p.NextCursor, testFingerprint, 1); !errors.Is(err, ErrCursorExpired) {
		t.Fatal(err)
	}
	p = putTestPage(t, &s)
	end := s.BeginMutation()
	end()
	if err := s.Release(p.NextCursor, testFingerprint); !errors.Is(err, ErrInvalidated) {
		t.Fatal(err)
	}
	s.Close()
	s.Close()
	if _, err := s.Start(ctx); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if _, err := s.Continue(ctx, p.NextCursor, testFingerprint, 1); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if err := s.Release(p.NextCursor, testFingerprint); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if err := s.Configure(Config{}); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}

func TestTicketLimitsCancellationAndEntropy(t *testing.T) {
	ctx := context.Background()
	var s Store
	if s.Config() != DefaultConfig() {
		t.Fatal("zero defaults")
	}
	ticket, err := s.Start(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ticket.Config() != s.Config() {
		t.Fatal("ticket policy")
	}
	if _, err := s.Start(ctx); !errors.Is(err, ErrCapacity) {
		t.Fatal(err)
	}
	if err := ticket.Check(ctx); err != nil {
		t.Fatal(err)
	}
	end := s.BeginMutation()
	inner := s.BeginMutation()
	end()
	if err := ticket.Check(ctx); !errors.Is(err, ErrInvalidated) {
		t.Fatal(err)
	}
	inner()
	ticket.Abort()
	ticket.Abort()
	var nilTicket *Ticket
	nilTicket.Abort()
	if err := ticket.Check(ctx); !errors.Is(err, ErrTicketClosed) {
		t.Fatal(err)
	}
	if _, err := ticket.Commit(ctx, testFingerprint, testPopulation(), 1); !errors.Is(err, ErrTicketClosed) {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := s.Start(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	ticket, _ = s.Start(ctx)
	if err := ticket.Check(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := ticket.Commit(cancelled, testFingerprint, testPopulation(), 1); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	p := putTestPage(t, &s)
	if _, err := s.Continue(cancelled, p.NextCursor, testFingerprint, 1); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	for _, size := range []int{0, 501} {
		ticket, _ = s.Start(ctx)
		if _, err := ticket.Commit(ctx, testFingerprint, testPopulation(), size); !errors.Is(err, ErrInvalidRequest) {
			t.Fatal(err)
		}
	}
	ticket, _ = s.Start(ctx)
	s.Close()
	if err := ticket.Check(ctx); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if _, err := ticket.Commit(ctx, testFingerprint, testPopulation(), 1); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	var entropyFailure Store
	entropyFailure.entropy = bytes.NewReader(nil)
	if _, err := entropyFailure.Start(ctx); !errors.Is(err, io.EOF) {
		t.Fatal(err)
	}
	entropyFailure.entropy = bytes.NewReader(make([]byte, 32))
	ticket, _ = entropyFailure.Start(ctx)
	if _, err := ticket.Commit(ctx, testFingerprint, testPopulation(), 1); !errors.Is(err, io.EOF) {
		t.Fatal(err)
	}
	var collision Store
	collision.entropy = bytes.NewReader(make([]byte, 256))
	putTestPage(t, &collision)
	ticket, _ = collision.Start(ctx)
	if _, err := ticket.Commit(ctx, testFingerprint, testPopulation(), 1); !errors.Is(err, ErrCapacity) {
		t.Fatal(err)
	}
}

func TestConfigAndPopulationValidation(t *testing.T) {
	ctx := context.Background()
	for _, c := range []Config{{TTL: -1}, {MaxSessions: -1}, {MaxResults: -1}, {MaxBytes: -1}, {MaxBuildBytes: -1}, {MaxBuilds: -1}, {MaxPageSize: -1}, {MaxCandidates: -1}, {MaxScannedNodes: -1}} {
		var s Store
		if err := s.Configure(c); !errors.Is(err, ErrInvalidRequest) {
			t.Fatal(err)
		}
	}
	custom := Config{TTL: time.Second, MaxSessions: 1, MaxResults: 9, MaxBytes: 100000, MaxBuildBytes: 100000, MaxBuilds: 2, MaxPageSize: 10, MaxCandidates: 2, MaxScannedNodes: 20}
	if actual, err := NormalizeConfig(custom); err != nil || actual != custom {
		t.Fatal(actual, err)
	}
	var s Store
	page := putTestPage(t, &s)
	ticket, _ := s.Start(ctx)
	if err := s.Configure(custom); err != nil {
		t.Fatal(err)
	}
	if err := ticket.Check(ctx); !errors.Is(err, ErrInvalidated) {
		t.Fatal(err)
	}
	ticket.Abort()
	if _, err := s.Continue(ctx, page.NextCursor, testFingerprint, 1); !errors.Is(err, ErrInvalidated) {
		t.Fatal(err)
	}
	putTestPage(t, &s)
	ticket, _ = s.Start(ctx)
	if _, err := ticket.Commit(ctx, testFingerprint, testPopulation(), 1); !errors.Is(err, ErrCapacity) {
		t.Fatal(err)
	}
	// Invalid population must never be published.
	bad := []Population{{Mode: "bad"}, {Mode: IDOnly, RankedCount: 1}, {Mode: RankedOnly, Hits: []Hit{{ID: "a"}}}, {Mode: IDOnly, CandidateLimit: -1}}
	for _, change := range []func(*Population){
		func(p *Population) { p.Hits[0].ID = "" }, func(p *Population) { p.Hits[0].ID = "\xff" },
		func(p *Population) { p.Hits[0].Phase = RankedPhase }, func(p *Population) { p.Hits[1].ID = "a" },
		func(p *Population) { p.Grouped = true }, func(p *Population) { p.Hits[0].GroupKey = "A" },
		func(p *Population) { p.Hits[0].Similarity = math.Inf(1) }, func(p *Population) { p.Hits[0].RRFScore = math.NaN() },
	} {
		p := testPopulation()
		change(&p)
		bad = append(bad, p)
	}
	p := testPopulation()
	p.Grouped = true
	for i := range p.Hits {
		p.Hits[i].GroupKey = "same"
	}
	bad = append(bad, p)
	for i, p := range bad {
		if _, _, err := clonePopulation(ctx, p, DefaultConfig()); !errors.Is(err, ErrInvalidRequest) {
			t.Fatalf("%d: %v", i, err)
		}
	}
	for _, c := range []Config{{MaxResults: 1}, {MaxBuildBytes: 1}, {MaxBytes: 1}, {MaxBuildBytes: 300}, {MaxBytes: 300}} {
		normal, _ := NormalizeConfig(c)
		if _, _, err := clonePopulation(ctx, testPopulation(), normal); !errors.Is(err, ErrCapacity) {
			t.Fatal(err)
		}
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, _, err := clonePopulation(cancelled, testPopulation(), DefaultConfig()); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	var tight Store
	_ = tight.Configure(Config{MaxBytes: 1500})
	putTestPage(t, &tight)
	ticket, _ = tight.Start(ctx)
	if _, err := ticket.Commit(ctx, testFingerprint, testPopulation(), 1); !errors.Is(err, ErrCapacity) {
		t.Fatal(err)
	}
}

func TestConcurrentReplaysAndMutations(t *testing.T) {
	ctx := context.Background()
	var s Store
	page := putTestPage(t, &s)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				p, err := s.Continue(ctx, page.NextCursor, testFingerprint, 1)
				if err != nil && !errors.Is(err, ErrInvalidated) {
					t.Error(err)
				}
				if p != nil && p.Results[0].ID != "b" {
					t.Error("torn page")
				}
			}
		}()
	}
	for i := 0; i < 20; i++ {
		done := s.BeginMutation()
		done()
	}
	wg.Wait()
}

func FuzzCursorDecoder(f *testing.F) {
	f.Add("")
	f.Add(strings.Repeat("A", 87))
	f.Add("not-a-cursor")
	f.Fuzz(func(t *testing.T, token string) {
		var s Store
		s.keyReady = true
		s.mu.Lock()
		_, _, _, _ = s.decodeCursorLocked(token)
		s.mu.Unlock()
	})
}
