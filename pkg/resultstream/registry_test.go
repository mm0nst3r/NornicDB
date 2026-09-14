package resultstream

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestRegistryProgressesBeyondInitialPopulation(t *testing.T) {
	var expansions atomic.Int32
	stream, err := NewProgressive([][]any{{0}, {1}}, false, 2, func(_ context.Context, depth int) ([][]any, bool, error) {
		expansions.Add(1)
		if depth > 9 {
			depth = 9
		}
		rows := make([][]any, depth)
		for index := range rows {
			rows[index] = []any{index}
		}
		return rows, depth >= 9, nil
	})
	require.NoError(t, err)

	registry, err := NewRegistry(Config{TTL: time.Minute, MaxStreams: 4, MaxPageSize: 4})
	require.NoError(t, err)
	t.Cleanup(registry.Close)

	scope := Scope{Owner: "alice", Database: "nornic"}
	first, err := registry.Start(context.Background(), scope, stream, 2)
	require.NoError(t, err)
	require.Equal(t, [][]any{{0}, {1}}, first.Rows)
	require.True(t, first.HasMore)
	require.NotEmpty(t, first.QID)

	second, err := registry.Pull(context.Background(), scope, first.QID, 3)
	require.NoError(t, err)
	require.Equal(t, [][]any{{2}, {3}, {4}}, second.Rows)
	require.True(t, second.HasMore)

	replayed, err := registry.Pull(context.Background(), scope, first.QID, 3)
	require.NoError(t, err)
	require.Equal(t, second.Rows, replayed.Rows)
	require.Equal(t, second.QID, replayed.QID)

	qid := second.QID
	var all [][]any
	for qid != "" {
		page, pullErr := registry.Pull(context.Background(), scope, qid, 4)
		require.NoError(t, pullErr)
		all = append(all, page.Rows...)
		qid = page.QID
	}
	require.Equal(t, [][]any{{5}, {6}, {7}, {8}}, all)
	require.GreaterOrEqual(t, expansions.Load(), int32(2))
}

func TestRegistryBindsScopeAndDiscardsStream(t *testing.T) {
	stream, err := NewProgressive([][]any{{"first"}, {"second"}}, false, 2, func(_ context.Context, _ int) ([][]any, bool, error) {
		return [][]any{{"first"}, {"second"}, {"third"}}, true, nil
	})
	require.NoError(t, err)

	registry, err := NewRegistry(Config{TTL: time.Minute, MaxStreams: 4, MaxPageSize: 2})
	require.NoError(t, err)
	t.Cleanup(registry.Close)

	scope := Scope{Owner: "alice", Database: "nornic"}
	first, err := registry.Start(context.Background(), scope, stream, 1)
	require.NoError(t, err)

	_, err = registry.Pull(context.Background(), Scope{Owner: "bob", Database: "nornic"}, first.QID, 1)
	require.ErrorIs(t, err, ErrInvalidQID)

	require.NoError(t, registry.Discard(scope, first.QID))
	_, err = registry.Pull(context.Background(), scope, first.QID, 1)
	require.True(t, errors.Is(err, ErrInvalidQID) || errors.Is(err, ErrExpiredQID))
}

func TestScopeDigestIsKeyedAndBindsOwnerAndDatabase(t *testing.T) {
	var firstSecret [32]byte
	var secondSecret [32]byte
	firstSecret[0] = 1
	secondSecret[0] = 2
	scope := Scope{Owner: "sub:alice", Database: "nornic"}

	require.Equal(t, scopeDigest(firstSecret, scope), scopeDigest(firstSecret, scope))
	require.NotEqual(t, scopeDigest(firstSecret, scope), scopeDigest(secondSecret, scope))
	require.NotEqual(t, scopeDigest(firstSecret, scope), scopeDigest(firstSecret, Scope{Owner: "sub:bob", Database: "nornic"}))
	require.NotEqual(t, scopeDigest(firstSecret, scope), scopeDigest(firstSecret, Scope{Owner: "sub:alice", Database: "other"}))
}

func TestTokenMACMatchesStandardHMACSHA256(t *testing.T) {
	var secret [32]byte
	for index := range secret {
		secret[index] = byte(index + 1)
	}
	body := make([]byte, tokenBodyBytes)
	for index := range body {
		body[index] = byte(index * 3)
	}

	standard := hmac.New(sha256.New, secret[:])
	_, err := standard.Write(body)
	require.NoError(t, err)
	actual := tokenMAC(secret, body)
	require.Equal(t, standard.Sum(nil), actual[:])
}
