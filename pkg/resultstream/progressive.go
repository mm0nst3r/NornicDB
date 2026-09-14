package resultstream

import (
	"context"
	"sync"
)

// ExpandFunc returns the complete ranked prefix available at depth. The
// returned prefix must preserve every previously returned row at its index.
type ExpandFunc func(ctx context.Context, depth int) (rows [][]any, exhausted bool, err error)

// Progressive grows a replayable row prefix only when buffered rows cannot
// satisfy a pull. Expansion work runs without registry locks.
type Progressive struct {
	mu        sync.Mutex
	rows      [][]any
	depth     int
	exhausted bool
	closed    bool
	expanding chan struct{}
	expand    ExpandFunc
}

// NewProgressive constructs a progressively expandable stream.
func NewProgressive(initial [][]any, exhausted bool, initialDepth int, expand ExpandFunc) (*Progressive, error) {
	if initialDepth < len(initial) || (!exhausted && expand == nil) {
		return nil, ErrInvalidPosition
	}
	return &Progressive{
		rows:      initial,
		depth:     max(initialDepth, 1),
		exhausted: exhausted,
		expand:    expand,
	}, nil
}

// Pull returns at most n rows and performs enough geometric expansion to know
// whether another row exists.
func (s *Progressive) Pull(ctx context.Context, position uint64, n int) (*Page, error) {
	if n <= 0 {
		return nil, ErrInvalidPageSize
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		s.mu.Lock()
		if s.closed {
			s.mu.Unlock()
			return nil, ErrClosed
		}
		if position > uint64(len(s.rows)) {
			s.mu.Unlock()
			return nil, ErrInvalidPosition
		}
		end := position + uint64(n)
		if end > uint64(len(s.rows)) {
			end = uint64(len(s.rows))
		}
		needLookahead := end == uint64(len(s.rows)) && !s.exhausted
		if !needLookahead {
			page := s.pageLocked(position, end)
			s.mu.Unlock()
			return page, nil
		}
		if s.expanding != nil {
			done := s.expanding
			s.mu.Unlock()
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-done:
				continue
			}
		}

		target := s.depth * 2
		minimum := int(end) + 1
		if target < minimum {
			target = minimum
		}
		done := make(chan struct{})
		s.expanding = done
		expand := s.expand
		s.mu.Unlock()

		rows, exhausted, err := expand(ctx, target)

		s.mu.Lock()
		if err == nil {
			if len(rows) < len(s.rows) {
				err = ErrInvalidPosition
			} else {
				s.rows = rows
				s.depth = target
				s.exhausted = exhausted
			}
		}
		s.expanding = nil
		close(done)
		s.mu.Unlock()
		if err != nil {
			return nil, err
		}
	}
}

func (s *Progressive) pageLocked(position, end uint64) *Page {
	hasMore := end < uint64(len(s.rows))
	page := &Page{
		Rows:     s.rows[position:end],
		Position: position,
		Next:     end,
		HasMore:  hasMore,
	}
	if s.exhausted {
		total := uint64(len(s.rows))
		page.Total = &total
	}
	return page
}

// Close releases references retained by the stream.
func (s *Progressive) Close() error {
	s.mu.Lock()
	s.closed = true
	s.rows = nil
	s.expand = nil
	s.mu.Unlock()
	return nil
}
