package search

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/orneryd/nornicdb/pkg/resultstream"
	"github.com/orneryd/nornicdb/pkg/storage"
)

type graphMutationVersionProvider interface {
	GraphMutationVersion() (uint64, bool)
}

// CompleteContinuationPolicy bounds exact population materialization. A zero
// field uses its default so callers can override one limit independently.
type CompleteContinuationPolicy struct {
	MaxScannedNodes     int
	MaxMembers          int
	MaxPassages         int
	MaxBuildBytes       int64
	MaxBuildDuration    time.Duration
	MaxConcurrentBuilds int64
}

func defaultCompleteContinuationPolicy() CompleteContinuationPolicy {
	return CompleteContinuationPolicy{
		MaxScannedNodes:     1_000_000,
		MaxMembers:          100_000,
		MaxPassages:         1_000_000,
		MaxBuildBytes:       256 << 20,
		MaxBuildDuration:    30 * time.Second,
		MaxConcurrentBuilds: 1,
	}
}

// SetCompleteContinuationPolicy replaces the limits used by future complete
// continuation builds. Zero fields retain their defaults.
func (s *Service) SetCompleteContinuationPolicy(policy CompleteContinuationPolicy) {
	defaults := defaultCompleteContinuationPolicy()
	if policy.MaxScannedNodes <= 0 {
		policy.MaxScannedNodes = defaults.MaxScannedNodes
	}
	if policy.MaxMembers <= 0 {
		policy.MaxMembers = defaults.MaxMembers
	}
	if policy.MaxPassages <= 0 {
		policy.MaxPassages = defaults.MaxPassages
	}
	if policy.MaxBuildBytes <= 0 {
		policy.MaxBuildBytes = defaults.MaxBuildBytes
	}
	if policy.MaxBuildDuration <= 0 {
		policy.MaxBuildDuration = defaults.MaxBuildDuration
	}
	if policy.MaxConcurrentBuilds <= 0 {
		policy.MaxConcurrentBuilds = defaults.MaxConcurrentBuilds
	}
	s.completePolicyMu.Lock()
	s.completePolicy.Store(policy)
	s.completePolicyGen.Add(1)
	s.completePolicyMu.Unlock()
}

func (s *Service) acquireCompleteContinuationBuild() (CompleteContinuationPolicy, uint64, bool) {
	s.completePolicyMu.RLock()
	defer s.completePolicyMu.RUnlock()
	policy, ok := s.completePolicy.Load().(CompleteContinuationPolicy)
	if !ok {
		policy = defaultCompleteContinuationPolicy()
	}
	policyID := s.completePolicyGen.Load()
	for {
		active := s.completeBuilds.Load()
		if active >= policy.MaxConcurrentBuilds {
			return policy, policyID, false
		}
		if s.completeBuilds.CompareAndSwap(active, active+1) {
			return policy, policyID, true
		}
	}
}

type catalogContinuationStream struct {
	mu       sync.RWMutex
	results  []SearchResult
	engine   storage.Engine
	version  uint64
	policy   *atomic.Uint64
	policyID uint64
	metadata map[string]any
	closed   bool
}

type continuationGroup struct {
	ranked  []SearchResult
	catalog []SearchResult
}

func (s *Service) newIDContinuationStream(ctx context.Context, options SearchOptions, request SearchContinuationRequest) (resultstream.Stream, error) {
	return s.newCompleteContinuationStream(ctx, options, request, nil)
}

func (s *Service) newCompleteContinuationStream(ctx context.Context, options SearchOptions, request SearchContinuationRequest, ranked *SearchResponse) (resultstream.Stream, error) {
	started := time.Now()
	policy, policyID, admitted := s.acquireCompleteContinuationBuild()
	if !admitted {
		return nil, resultstream.ErrCapacity
	}
	defer s.completeBuilds.Add(-1)

	provider, ok := s.engine.(graphMutationVersionProvider)
	if !ok {
		return nil, fmt.Errorf("complete continuation requires graph mutation revisions: %w", resultstream.ErrInvalidated)
	}
	version, supported := provider.GraphMutationVersion()
	if !supported {
		return nil, fmt.Errorf("complete continuation requires graph mutation revisions: %w", resultstream.ErrInvalidated)
	}

	s.mu.RLock()
	decayFilter := s.nodeDecayFilter
	s.mu.RUnlock()
	rankedByID := make(map[string]SearchResult)
	searchMethod := "id"
	fallbackTriggered := false
	if ranked != nil {
		searchMethod = ranked.SearchMethod
		fallbackTriggered = ranked.FallbackTriggered
		for _, result := range ranked.Results {
			result.Phase = SearchContinuationRankedPhase
			rankedByID[searchResultID(result)] = result
		}
	}
	members := make(map[string]*continuationGroup)
	scannedNodes := 0
	passageCount := 0
	var retainedBytes int64
	err := storage.StreamNodesWithFallback(ctx, s.engine, 1000, func(node *storage.Node) error {
		if time.Since(started) > policy.MaxBuildDuration {
			return resultstream.ErrCapacity
		}
		scannedNodes++
		if scannedNodes > policy.MaxScannedNodes {
			return resultstream.ErrCapacity
		}
		eligible, err := s.continuationNodeEligible(node, &options, decayFilter)
		if err != nil || !eligible {
			return err
		}
		groupKey := ""
		if request.GroupBy != "" {
			value, valid := node.Properties[request.GroupBy].(string)
			if !valid || value == "" || !utf8.ValidString(value) {
				return fmt.Errorf("group property %q must be a nonempty UTF-8 string", request.GroupBy)
			}
			groupKey = value
		}
		id := string(node.ID)
		logicalID := id
		if groupKey != "" {
			logicalID = groupKey
		}
		candidate := SearchResult{ID: id, NodeID: node.ID}
		if selected, exists := rankedByID[id]; exists {
			candidate = compactContinuationResult(selected)
			candidate.ID = id
			candidate.NodeID = node.ID
			candidate.Phase = SearchContinuationRankedPhase
		} else {
			candidate.Phase = SearchContinuationCatalogPhase
		}
		candidate.GroupKey = groupKey
		retainedBytes += compactContinuationResultBytes(candidate)
		if retainedBytes > policy.MaxBuildBytes {
			return resultstream.ErrCapacity
		}
		group := members[logicalID]
		if group == nil {
			if len(members) >= policy.MaxMembers {
				return resultstream.ErrCapacity
			}
			group = &continuationGroup{}
			members[logicalID] = group
		}
		passageCount++
		if passageCount > policy.MaxPassages {
			return resultstream.ErrCapacity
		}
		if candidate.Phase == SearchContinuationRankedPhase {
			group.ranked = append(group.ranked, candidate)
		} else {
			group.catalog = append(group.catalog, candidate)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if time.Since(started) > policy.MaxBuildDuration {
		return nil, resultstream.ErrCapacity
	}
	after, supported := provider.GraphMutationVersion()
	if !supported || after != version {
		return nil, resultstream.ErrInvalidated
	}
	if request.MaxResults > 0 && len(members) > request.MaxResults {
		return nil, resultstream.ErrCapacity
	}
	results := make([]SearchResult, 0, len(members))
	for _, group := range members {
		passages := group.catalog
		if len(group.ranked) > 0 {
			passages = group.ranked
			sort.Slice(passages, func(i, j int) bool {
				return betterContinuationRepresentative(passages[i], passages[j])
			})
		} else {
			sort.Slice(passages, func(i, j int) bool { return passages[i].ID < passages[j].ID })
		}
		representative := passages[0]
		if request.GroupBy != "" {
			representative.Passages = make([]SearchPassage, len(passages))
			for index := range passages {
				representative.Passages[index] = searchPassageFromResult(passages[index])
			}
		}
		results = append(results, representative)
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Phase != results[j].Phase {
			return results[i].Phase == SearchContinuationRankedPhase
		}
		if results[i].Phase == SearchContinuationRankedPhase && results[i].Score != results[j].Score {
			return results[i].Score > results[j].Score
		}
		left, right := results[i].ID, results[j].ID
		if request.GroupBy != "" {
			left, right = results[i].GroupKey, results[j].GroupKey
		}
		if left != right {
			return left < right
		}
		return results[i].ID < results[j].ID
	})
	eligibleCount := len(results)
	rankedCount := 0
	for index := range results {
		if results[index].Phase == SearchContinuationRankedPhase {
			rankedCount++
		}
	}
	if s.completePolicyGen.Load() != policyID {
		return nil, resultstream.ErrInvalidated
	}
	return &catalogContinuationStream{
		results:  results,
		engine:   s.engine,
		version:  version,
		policy:   &s.completePolicyGen,
		policyID: policyID,
		metadata: map[string]any{
			"search_method":         searchMethod,
			"fallback_triggered":    fallbackTriggered,
			"discovered":            eligibleCount,
			"mode":                  request.Mode,
			"ranked_count":          rankedCount,
			"eligible_count":        &eligibleCount,
			"ranked_pool_exhausted": true,
		},
	}, nil
}

func compactContinuationResultBytes(result SearchResult) int64 {
	const fixedBytes = int64(8*5 + 8*4)
	return fixedBytes + int64(len(result.ID)+len(result.NodeID)+len(result.GroupKey)+len(result.Phase))
}

func compactContinuationResult(result SearchResult) SearchResult {
	return SearchResult{
		ID: result.ID, NodeID: result.NodeID, GroupKey: result.GroupKey, Phase: result.Phase,
		Score: result.Score, Similarity: result.Similarity, RRFScore: result.RRFScore,
		VectorRank: result.VectorRank, BM25Rank: result.BM25Rank,
	}
}

func hydrateContinuationResult(ranked, stored SearchResult) SearchResult {
	ranked.NodeID = stored.NodeID
	ranked.Type = stored.Type
	ranked.Labels = stored.Labels
	ranked.Title = stored.Title
	ranked.Description = stored.Description
	ranked.ContentPreview = stored.ContentPreview
	ranked.Properties = stored.Properties
	return ranked
}

func searchPassageFromResult(result SearchResult) SearchPassage {
	return SearchPassage{
		ID: result.ID, NodeID: result.NodeID, Phase: result.Phase, Type: result.Type,
		Labels: result.Labels, Title: result.Title, Description: result.Description,
		ContentPreview: result.ContentPreview, Properties: result.Properties,
		Score: result.Score, Similarity: result.Similarity, RRFScore: result.RRFScore,
		VectorRank: result.VectorRank, BM25Rank: result.BM25Rank,
	}
}

func betterContinuationRepresentative(candidate, current SearchResult) bool {
	if candidate.Score != current.Score {
		return candidate.Score > current.Score
	}
	if candidate.ID != current.ID {
		return candidate.ID < current.ID
	}
	if candidate.RRFScore != current.RRFScore {
		return candidate.RRFScore > current.RRFScore
	}
	if candidate.Similarity != current.Similarity {
		return candidate.Similarity > current.Similarity
	}
	if candidate.VectorRank != current.VectorRank {
		return candidate.VectorRank < current.VectorRank
	}
	return candidate.BM25Rank < current.BM25Rank
}

func (s *Service) continuationNodeEligible(node *storage.Node, options *SearchOptions, decayFilter NodeDecayFilterFunc) (bool, error) {
	if node == nil || node.VisibilitySuppressed || (decayFilter != nil && decayFilter(string(node.ID))) {
		return false, nil
	}
	if len(options.Types) > 0 {
		matched := false
		for _, wanted := range options.Types {
			wanted = strings.ToLower(wanted)
			for _, label := range node.Labels {
				if wanted == strings.ToLower(label) {
					matched = true
					break
				}
			}
			if nodeType, ok := node.Properties["type"].(string); ok && wanted == strings.ToLower(nodeType) {
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
	if !nodeMatchesFilters(node, options.Filters) {
		return false, nil
	}
	return s.shouldIndexNode(node)
}

func searchResultFromContinuationNode(node *storage.Node) SearchResult {
	result := SearchResult{
		ID:         string(node.ID),
		NodeID:     node.ID,
		Labels:     append([]string(nil), node.Labels...),
		Properties: node.Properties,
	}
	if len(node.Labels) > 0 {
		result.Type = node.Labels[0]
	}
	return result
}

func (s *catalogContinuationStream) Pull(ctx context.Context, position uint64, n int) (*resultstream.Page, error) {
	if n <= 0 {
		return nil, resultstream.ErrInvalidPageSize
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	provider, ok := s.engine.(graphMutationVersionProvider)
	if !ok {
		return nil, resultstream.ErrInvalidated
	}
	version, supported := provider.GraphMutationVersion()
	if !supported || version != s.version {
		return nil, resultstream.ErrInvalidated
	}
	if s.policy == nil || s.policy.Load() != s.policyID {
		return nil, resultstream.ErrInvalidated
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return nil, resultstream.ErrClosed
	}
	if position > uint64(len(s.results)) {
		return nil, resultstream.ErrInvalidPosition
	}
	end := min(position+uint64(n), uint64(len(s.results)))
	hasMore := end < uint64(len(s.results))
	total := uint64(len(s.results))
	results, err := hydrateContinuationResults(s.engine, s.results[position:end])
	if err != nil {
		return nil, err
	}
	metadata := make(map[string]any, len(s.metadata)+2)
	for key, value := range s.metadata {
		metadata[key] = value
	}
	metadata["collection_exhausted"] = !hasMore
	if hasMore {
		metadata["completion"] = SearchContinuationMoreResults
	} else {
		metadata["completion"] = SearchContinuationCollectionComplete
	}
	return &resultstream.Page{
		Rows:     continuationRows(results),
		Position: position,
		Next:     end,
		HasMore:  hasMore,
		Total:    &total,
		Metadata: metadata,
	}, nil
}

func hydrateContinuationResults(engine storage.Engine, compact []SearchResult) ([]SearchResult, error) {
	ids := make([]storage.NodeID, 0, len(compact))
	seen := make(map[storage.NodeID]struct{}, len(compact))
	for index := range compact {
		if _, exists := seen[compact[index].NodeID]; !exists {
			seen[compact[index].NodeID] = struct{}{}
			ids = append(ids, compact[index].NodeID)
		}
		for passageIndex := range compact[index].Passages {
			id := compact[index].Passages[passageIndex].NodeID
			if _, exists := seen[id]; !exists {
				seen[id] = struct{}{}
				ids = append(ids, id)
			}
		}
	}
	nodes, err := engine.BatchGetNodes(ids)
	if err != nil {
		return nil, err
	}
	results := make([]SearchResult, len(compact))
	for index := range compact {
		node := nodes[compact[index].NodeID]
		if node == nil {
			return nil, resultstream.ErrInvalidated
		}
		results[index] = hydrateContinuationResult(compact[index], searchResultFromContinuationNode(node))
		if len(compact[index].Passages) == 0 {
			continue
		}
		results[index].Passages = make([]SearchPassage, len(compact[index].Passages))
		for passageIndex := range compact[index].Passages {
			passage := compact[index].Passages[passageIndex]
			passageNode := nodes[passage.NodeID]
			if passageNode == nil {
				return nil, resultstream.ErrInvalidated
			}
			passageResult := SearchResult{
				ID: passage.ID, NodeID: passage.NodeID, Phase: passage.Phase,
				Score: passage.Score, Similarity: passage.Similarity, RRFScore: passage.RRFScore,
				VectorRank: passage.VectorRank, BM25Rank: passage.BM25Rank,
			}
			passageResult = hydrateContinuationResult(passageResult, searchResultFromContinuationNode(passageNode))
			results[index].Passages[passageIndex] = searchPassageFromResult(passageResult)
		}
	}
	return results, nil
}

func (s *catalogContinuationStream) Close() error {
	s.mu.Lock()
	s.closed = true
	s.results = nil
	s.mu.Unlock()
	return nil
}
