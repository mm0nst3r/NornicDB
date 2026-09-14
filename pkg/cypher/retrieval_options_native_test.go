package cypher

import (
	"testing"

	"github.com/orneryd/nornicdb/pkg/search"
	"github.com/stretchr/testify/require"
)

func TestRetrievalOptionsNativeOverrides(t *testing.T) {
	for _, keys := range [][2]string{{"rerankTruncation", "rerankFailurePolicy"}, {"rerank_truncation", "rerank_failure_policy"}} {
		for _, truncation := range []bool{false, true} {
			for _, policy := range []string{"error", "original"} {
				opts, _, err := retrievalOptions("document", map[string]interface{}{keys[0]: truncation, keys[1]: policy}, true)
				require.NoError(t, err)
				require.NotNil(t, opts.RerankTruncation)
				require.Equal(t, truncation, *opts.RerankTruncation)
				require.Equal(t, search.RerankFailurePolicy(policy), opts.RerankFailurePolicy)
				require.True(t, opts.RerankEnabled)
			}
		}
		for _, req := range []map[string]interface{}{{keys[0]: "false"}, {keys[1]: "ignore"}, {keys[1]: true}} {
			_, _, err := retrievalOptions("document", req, true)
			require.Error(t, err)
		}
	}
	opts, _, err := retrievalOptions("document", nil, false)
	require.NoError(t, err)
	require.Nil(t, opts.RerankTruncation)
	require.Empty(t, opts.RerankFailurePolicy)
}
