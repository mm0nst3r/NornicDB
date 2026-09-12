package nornicgrpc

import (
	"context"
	"testing"

	gen "github.com/orneryd/nornicdb/pkg/nornicgrpc/gen"
	"github.com/orneryd/nornicdb/pkg/search"
	"github.com/orneryd/nornicdb/pkg/voyage"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type failedVoyageSearch struct{ err error }

func (s failedVoyageSearch) Search(context.Context, string, []float32, *search.SearchOptions) (*search.SearchResponse, error) {
	return nil, s.err
}

func TestVoyageFailureDiagnosticsReachGRPCStatus(t *testing.T) {
	failure := &voyage.Error{Kind: voyage.MalformedResponse, Metadata: voyage.Metadata{
		StatusCode: 200, RequestID: "grpc-failed-request", Attempts: 1,
		Usage: &voyage.Usage{TotalTokens: 987},
	}}
	service, err := NewService(Config{}, nil, nil, failedVoyageSearch{failure})
	require.NoError(t, err)
	result, err := service.SearchText(context.Background(), &gen.SearchTextRequest{Query: "document query"})
	require.Nil(t, result)
	require.Equal(t, codes.Internal, status.Code(err))
	message := status.Convert(err).Message()
	require.Contains(t, message, "grpc-failed-request")
	require.Contains(t, message, `"attempts":1`)
	require.Contains(t, message, `"total_tokens":987`)
}
