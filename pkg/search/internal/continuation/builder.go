package continuation

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// Builder collapses eligible passages/frames into distinct parents before
// pagination. Only one representative per logical member is retained.
type Builder struct {
	mode     Mode
	grouped  bool
	config   Config
	ranked   map[string]Hit
	members  map[string]Hit
	bytes    int64
	scanned  int
	finished bool
}

// NewBuilder snapshots a bounded ranked pool. Add supplies authoritative
// eligible membership; candidates not supplied to Add cannot enter the result.
func NewBuilder(mode Mode, grouped bool, ranked []Hit, config Config) (*Builder, error) {
	c, err := NormalizeConfig(config)
	if err != nil {
		return nil, err
	}
	if !validMode(mode) || (mode == IDOnly && len(ranked) > 0) {
		return nil, ErrInvalidRequest
	}
	if len(ranked) > c.MaxCandidates {
		return nil, ErrCapacity
	}
	b := &Builder{mode: mode, grouped: grouped, config: c, ranked: make(map[string]Hit), members: make(map[string]Hit)}
	for _, h := range ranked {
		if h.ID == "" || !utf8.ValidString(h.ID) || !finiteHit(h) {
			return nil, ErrInvalidRequest
		}
		h.GroupKey = ""
		h.Phase = RankedPhase
		old, exists := b.ranked[h.ID]
		if exists && !betterRepresentative(h, old) {
			continue
		}
		delta := hitBytes(h)
		if exists {
			delta -= hitBytes(old)
		}
		if delta > c.MaxBuildBytes-b.bytes {
			return nil, ErrCapacity
		}
		b.bytes += delta
		h.ID = strings.Clone(h.ID)
		b.ranked[h.ID] = h
	}
	return b, nil
}

// Add visits one metadata-eligible node. With grouping, group must be a
// nonempty scalar parent key; missing/invalid keys are errors, never omissions.
func (b *Builder) Add(id, group string) error {
	if b.finished {
		return ErrTicketClosed
	}
	b.scanned++
	if b.scanned > b.config.MaxScannedNodes {
		return ErrCapacity
	}
	if id == "" || !utf8.ValidString(id) || !utf8.ValidString(group) || (b.grouped && group == "") || (!b.grouped && group != "") {
		return fmt.Errorf("invalid node/group key: %w", ErrInvalidRequest)
	}
	ranked, hasRank := b.ranked[id]
	if b.mode == RankedOnly && !hasRank {
		return nil
	}
	key := id
	if b.grouped {
		key = group
	}
	incoming := Hit{ID: id, GroupKey: group, Phase: CatalogPhase}
	if hasRank {
		incoming = ranked
		incoming.GroupKey = group
	}
	old, exists := b.members[key]
	if exists {
		if old.Phase == RankedPhase && incoming.Phase != RankedPhase {
			return nil
		}
		if old.Phase == incoming.Phase && !betterRepresentative(incoming, old) {
			return nil
		}
	} else if len(b.members) >= b.config.MaxResults {
		return ErrCapacity
	}
	delta := hitBytes(incoming)
	if exists {
		delta -= hitBytes(old)
	}
	if delta > b.config.MaxBuildBytes-b.bytes {
		return ErrCapacity
	}
	b.bytes += delta
	incoming.ID = strings.Clone(incoming.ID)
	incoming.GroupKey = strings.Clone(incoming.GroupKey)
	if b.grouped {
		key = incoming.GroupKey
	} else {
		key = incoming.ID
	}
	b.members[key] = incoming
	return nil
}

// Finish sorts a fixed prefix by descending score / ascending logical ID, and
// the unscored remainder by ascending logical ID. For grouped hits, equal-score
// representatives are chosen by node ID, independent of scan/insertion order.
func (b *Builder) Finish(ctx context.Context) (Population, error) {
	if b.finished {
		return Population{}, ErrTicketClosed
	}
	b.finished = true
	defer func() { b.ranked = nil; b.members = nil }()
	out := Population{Mode: b.mode, Grouped: b.grouped, Hits: make([]Hit, 0, len(b.members))}
	for _, h := range b.members {
		if err := ctx.Err(); err != nil {
			return Population{}, err
		}
		out.Hits = append(out.Hits, h)
		if h.Phase == RankedPhase {
			out.RankedCount++
		}
	}
	sort.Slice(out.Hits, func(i, j int) bool {
		a, c := out.Hits[i], out.Hits[j]
		if a.Phase != c.Phase {
			return a.Phase == RankedPhase
		}
		if a.Phase == RankedPhase && a.Score != c.Score {
			return a.Score > c.Score
		}
		ak, ck := a.ID, c.ID
		if b.grouped {
			ak, ck = a.GroupKey, c.GroupKey
		}
		if ak != ck {
			return ak < ck
		}
		return a.ID < c.ID
	})
	if err := ctx.Err(); err != nil {
		return Population{}, err
	}
	return out, nil
}

func betterRepresentative(a, b Hit) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	if a.ID != b.ID {
		return a.ID < b.ID
	}
	// Defensive canonicalisation of duplicate input candidates also makes
	// their diagnostic fields independent of the upstream iteration order.
	if a.RRFScore != b.RRFScore {
		return a.RRFScore > b.RRFScore
	}
	if a.Similarity != b.Similarity {
		return a.Similarity > b.Similarity
	}
	if a.VectorRank != b.VectorRank {
		return a.VectorRank < b.VectorRank
	}
	return a.BM25Rank < b.BM25Rank
}
