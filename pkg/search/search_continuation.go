package search

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/orneryd/nornicdb/pkg/search/internal/continuation"
	"github.com/orneryd/nornicdb/pkg/storage"
)

// SearchPageMode declares whether pagination covers a bounded retrieval pool or
// a complete, metadata-filtered collection. It does not change ANN recall.
type SearchPageMode = continuation.Mode

const (
	// SearchPageRanked returns only the fixed, bounded retrieval result.
	SearchPageRanked = continuation.RankedOnly
	// SearchPageRankedThenID appends every other metadata-eligible member in ID
	// order. The tail is unscored and need not match the query or have a vector.
	SearchPageRankedThenID = continuation.RankedThenID
	// SearchPageID bypasses retrieval and browses the metadata-filtered collection.
	SearchPageID = continuation.IDOnly
)

// SearchPageHit contains an ID and ranking metadata, never a full node or vector.
// Phase="catalog" means unscored, NOT a zero-relevance search match. GroupKey is
// the distinct parent value when SearchPageOptions.GroupBy is set.
type SearchPageHit = continuation.Hit

// SearchPageResponse distinguishes exhausted selected candidates from exhausted
// eligible collection. EligibleCount is unknown for SearchPageRanked; Total is
// always the declared population size, not an ANN estimate of corpus size.
type SearchPageResponse = continuation.Page

// SearchContinuationConfig bounds cursor lifetime, retained descriptors, initial
// scans, and simultaneous materialisations. Zero fields use documented defaults.
type SearchContinuationConfig = continuation.Config

// Cursor errors support errors.Is. Invalidated, expired, or process-restarted
// sessions must be restarted; a cursor is not a durable storage snapshot.
var (
	ErrSearchPageRequest        = continuation.ErrInvalidRequest
	ErrSearchCursorInvalid      = continuation.ErrInvalidCursor
	ErrSearchCursorMismatch     = continuation.ErrCursorMismatch
	ErrSearchCursorExpired      = continuation.ErrCursorExpired
	ErrSearchCursorInvalidated  = continuation.ErrInvalidated
	ErrSearchContinuationLimit  = continuation.ErrCapacity
	ErrSearchContinuationClosed = continuation.ErrClosed
)

// SearchPageOptions configures continuation independently of retrieval depth.
// Repeat query, embedding, SearchOptions, mode, group, and scope on every call.
// PageSize may change between calls. Cursor is empty on the initial request.
type SearchPageOptions struct {
	Mode     SearchPageMode
	PageSize int
	Cursor   string
	// GroupBy is a flat property containing a nonempty string parent ID. Filters
	// apply to hit nodes, not to joined parent nodes. Missing/invalid keys fail.
	GroupBy string
	// Scope MUST be supplied by a trusted adapter, e.g. authenticated subject +
	// authorization-policy revision. It is not a substitute for authorization.
	// Scope is mandatory even for embedded callers (which may use a fixed name).
	Scope string
}

type searchContinuationState struct {
	store             continuation.Store
	policyMu          sync.Mutex // serializes master-flag no-op detection with flag swaps
	storageMu         sync.Mutex
	storageVersion    uint64
	storageVersionSet bool
}

// refreshSearchContinuationStorage notices completed writes even when indexing
// is deferred. Custom engines without revision reporting retain the documented
// explicit mutation-bracketing contract.
func (s *Service) refreshSearchContinuationStorage() bool {
	provider, ok := s.engine.(storage.GraphMutationVersionProvider)
	if !ok {
		return false
	}
	c := &s.continuation
	c.storageMu.Lock()
	defer c.storageMu.Unlock()
	version, supported := provider.GraphMutationVersion()
	if !supported {
		return false
	}
	changed := c.storageVersionSet && c.storageVersion != version
	if changed {
		c.store.BeginMutation()()
	}
	c.storageVersion, c.storageVersionSet = version, true
	return changed
}

// ConfigureSearchContinuation changes limits and invalidates outstanding cursors.
// Defaults: 5-minute fixed TTL, 32 sessions, 100,000 members/session, 64 MiB of
// retained descriptor accounting, 32 MiB/build, one concurrent build, 500/page,
// 5,000 ranked candidates, and 1,000,000 scanned nodes. Limits fail, not truncate.
func (s *Service) ConfigureSearchContinuation(config SearchContinuationConfig) error {
	if s == nil {
		return ErrSearchContinuationClosed
	}
	return s.continuation.store.Configure(config)
}

// BeginSearchContinuationMutation brackets writes/policy changes performed
// outside Service's indexing methods. Call BEFORE the write and defer its
// returned function until storage and indexing have BOTH finished. Nested use
// is safe. IndexNode/RemoveNode/BuildIndexes automatically bracket their own
// operations, but cannot cover an earlier direct storage write by the caller.
//
//	finish := svc.BeginSearchContinuationMutation()
//	defer finish()
//	if err := engine.UpdateNode(node); err != nil { return err }
//	return svc.IndexNode(node)
func (s *Service) BeginSearchContinuationMutation() func() {
	if s == nil {
		return func() {}
	}
	return s.continuation.store.BeginMutation()
}

// SearchPage retrieves once, freezes an ID/score population inside this service,
// and serves subsequent bounded pages without rerunning retrieval or rescanning.
// SearchOptions.Limit is the fixed ranked-pool depth, NOT the page size.
//
// SearchPageRankedThenID and SearchPageID perform one complete storage scan on
// the first request. State and scans are bounded by SearchContinuationConfig;
// exceeding a limit is an error, never a claim of complete enumeration. MMR is
// rejected because its sequence is not a descending score/ID order.
//
//	opts := DefaultSearchOptions()
//	opts.Limit = 200
//	opts.Filters = map[string][]string{"collection": {"summer"}}
//	pageOpts := &SearchPageOptions{Mode: SearchPageRankedThenID, PageSize: 50, Scope: "catalogue-reader"}
//	page, err := svc.SearchPage(ctx, "beach", embedding, opts, pageOpts)
//	if err != nil { return err }
//	pageOpts.Cursor = page.NextCursor // stop when NextCursor is empty
//
// This API returns descriptors only; fetch full nodes separately after applying
// normal authorization. Native Cypher exposes it through db.retrieve.page.
func (s *Service) SearchPage(ctx context.Context, query string, embedding []float32, opts *SearchOptions, paging *SearchPageOptions) (*SearchPageResponse, error) {
	if s == nil || s.engine == nil {
		return nil, ErrSearchContinuationClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.refreshSearchContinuationStorage()
	r, err := s.resolveSearchPageRequest(query, embedding, opts, paging)
	if err != nil {
		return nil, err
	}
	if r.page.Cursor != "" {
		page, err := s.continuation.store.Continue(ctx, r.page.Cursor, r.fingerprint, r.page.PageSize)
		if s.refreshSearchContinuationStorage() {
			return nil, ErrSearchCursorInvalidated
		}
		return page, err
	}
	// A lazy index build is itself a mutation. Finish it BEFORE reserving a
	// population epoch, otherwise the first request would invalidate itself.
	if r.page.Mode != SearchPageID {
		if err := s.EnsureWarm(ctx); err != nil {
			return nil, err
		}
	}
	s.refreshSearchContinuationStorage()
	ticket, err := s.continuation.store.Start(ctx)
	if err != nil {
		return nil, err
	}
	defer ticket.Abort()

	// Warmup may have reconfigured resource policy. Resolve again inside the
	// reserved epoch so the published cursor binds the policy actually used.
	// The first resolution was validation only and never modified caller data.
	r, err = s.resolveSearchPageRequest(query, embedding, opts, paging)
	if err != nil {
		return nil, err
	}
	if err := ticket.Check(ctx); err != nil {
		return nil, err
	}

	var ranked []continuation.Hit
	var retrieval *SearchResponse
	method, candidateLimit := "id", 0
	if r.page.Mode != SearchPageID {
		// Deliberately bypass the ordinary result cache: continuation binds its
		// own canonical SHA-256 request and must not inherit an older cached
		// response or rely on the ordinary cache's key/lifetime semantics.
		response, err := s.searchWithResultCache(ctx, r.query, r.embedding, &r.options, false)
		if err != nil {
			return nil, err
		}
		if response == nil {
			return nil, fmt.Errorf("nil retrieval response: %w", ErrSearchPageRequest)
		}
		if len(response.Results) > r.options.Limit {
			return nil, ErrSearchContinuationLimit
		}
		method = response.SearchMethod
		retrieval = response
		candidateLimit = resolveAdaptiveOverfetch(&r.options).maxLimit
		ranked = make([]continuation.Hit, 0, len(response.Results))
		for _, hit := range response.Results {
			ranked = append(ranked, continuation.Hit{ID: hit.ID, Score: hit.Score,
				Similarity: hit.Similarity, RRFScore: hit.RRFScore,
				VectorRank: hit.VectorRank, BM25Rank: hit.BM25Rank})
		}
	}
	if err := ticket.Check(ctx); err != nil {
		return nil, err
	}
	builder, err := continuation.NewBuilder(r.page.Mode, r.page.GroupBy != "", ranked, ticket.Config())
	if err != nil {
		return nil, err
	}
	s.mu.RLock()
	decayFilter := s.nodeDecayFilter
	s.mu.RUnlock()
	visited := 0
	visit := func(node *storage.Node) error {
		visited++ // count ALL scanned nodes, including metadata-filter rejects
		if visited > ticket.Config().MaxScannedNodes {
			return ErrSearchContinuationLimit
		}
		if err := ticket.Check(ctx); err != nil {
			return err
		}
		eligible, err := s.searchPageNodeEligible(node, &r.options, decayFilter)
		if err != nil {
			return err
		}
		if !eligible {
			return nil
		}
		group := ""
		if r.page.GroupBy != "" {
			var ok bool
			group, ok = node.Properties[r.page.GroupBy].(string)
			if !ok || group == "" {
				return fmt.Errorf("group property %q must be a nonempty string: %w", r.page.GroupBy, ErrSearchPageRequest)
			}
		}
		return builder.Add(string(node.ID), group)
	}
	if r.page.Mode == SearchPageRanked {
		for _, hit := range ranked {
			node, err := s.engine.GetNode(storage.NodeID(hit.ID))
			if err != nil {
				if errors.Is(err, storage.ErrNotFound) {
					return nil, ErrSearchCursorInvalidated
				}
				return nil, err
			}
			if err := visit(node); err != nil {
				return nil, err
			}
		}
	} else {
		if err := storage.StreamNodesWithFallback(ctx, s.engine, 1000, visit); err != nil {
			return nil, err
		}
	}
	population, err := builder.Finish(ctx)
	if err != nil {
		return nil, err
	}
	population.SearchMethod = method
	population.CandidateLimit = candidateLimit
	if retrieval != nil {
		population.TotalCandidates, population.FallbackTriggered = retrieval.TotalCandidates, retrieval.FallbackTriggered
		if metrics := retrieval.Metrics; metrics != nil {
			population.VectorStopReason, population.VectorCandidateLimit = metrics.VectorStopReason, metrics.VectorCandidateLimit
			population.BM25StopReason, population.BM25CandidateLimit = metrics.BM25StopReason, metrics.BM25CandidateLimit
			population.CandidateLimit = max(metrics.VectorCandidateLimit, metrics.BM25CandidateLimit)
		}
	}
	if s.refreshSearchContinuationStorage() {
		return nil, ErrSearchCursorInvalidated
	}
	page, err := ticket.Commit(ctx, r.fingerprint, population, r.page.PageSize)
	if s.refreshSearchContinuationStorage() {
		return nil, ErrSearchCursorInvalidated
	}
	return page, err
}

// ReleaseSearchPage frees a session before expiry. Pass the same request and
// scope with any issued cursor from that session. All its cursors then expire.
func (s *Service) ReleaseSearchPage(query string, embedding []float32, opts *SearchOptions, paging *SearchPageOptions) error {
	if s == nil {
		return ErrSearchContinuationClosed
	}
	r, err := s.resolveSearchPageRequest(query, embedding, opts, paging)
	if err != nil {
		return err
	}
	return s.continuation.store.Release(r.page.Cursor, r.fingerprint)
}

func (s *Service) searchPageNodeEligible(node *storage.Node, opts *SearchOptions, decay NodeDecayFilterFunc) (bool, error) {
	if node == nil {
		return false, fmt.Errorf("nil storage node: %w", ErrSearchPageRequest)
	}
	if node.VisibilitySuppressed || (decay != nil && decay(string(node.ID))) {
		return false, nil
	}
	if len(opts.Types) > 0 {
		matched := false
		for _, kind := range opts.Types {
			for _, label := range node.Labels {
				if kind == strings.ToLower(label) {
					matched = true
					break
				}
			}
			if kindValue, ok := node.Properties["type"].(string); ok && kind == strings.ToLower(kindValue) {
				matched = true
			}
			if matched {
				break
			}
		}
		if !matched {
			return false, nil
		}
	}
	if !nodeMatchesFilters(node, opts.Filters) {
		return false, nil
	}
	return s.shouldIndexNode(node)
}
