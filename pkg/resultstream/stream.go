// Package resultstream provides durable, position-addressed result streams.
package resultstream

import (
	"context"
	"errors"
	"time"
)

var (
	ErrClosed          = errors.New("result stream is closed")
	ErrInvalidPageSize = errors.New("invalid result stream page size")
	ErrInvalidPosition = errors.New("invalid result stream position")
	ErrInvalidated     = errors.New("result stream was invalidated by a data or policy change")
)

// Page is an immutable view of one stream position. Rows must be treated as
// read-only by adapters.
type Page struct {
	Rows      [][]any
	QID       string
	Position  uint64
	Next      uint64
	HasMore   bool
	Total     *uint64
	ExpiresAt time.Time
	Metadata  map[string]any
}

// Stream returns deterministic pages for explicit positions. Implementations
// must permit concurrent and repeated Pull calls.
type Stream interface {
	Pull(ctx context.Context, position uint64, n int) (*Page, error)
	Close() error
}
