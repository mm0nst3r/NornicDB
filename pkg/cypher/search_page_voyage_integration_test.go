package cypher

import (
	"encoding/json"
	"testing"

	"github.com/orneryd/nornicdb/pkg/search"
	"github.com/stretchr/testify/require"
)

func TestNativeSearchPageRetainsVoyageDetails(t *testing.T) {
	var page search.SearchPageResponse
	require.NoError(t, json.Unmarshal([]byte(`{"results":[{"id":"a","phase":"ranked","passages":[{"node_id":"a","chunk_index":1,"text":"complete supporting text","matched_by":["vector"]}]}],"rerank":{"provider":"voyage","status":"applied","submitted":101}}`), &page))
	wire := nativeSearchPage(&page)
	require.Equal(t, "applied", wire["rerank"].(map[string]any)["status"])
	hit := wire["results"].([]interface{})[0].(map[string]interface{})
	require.Equal(t, "complete supporting text", hit["passages"].([]any)[0].(map[string]any)["text"])
}

func TestRetrieveAndPageShareNativeRerankOptions(t *testing.T) {
	opts, _, err := retrievalOptions("query", map[string]interface{}{"rerankTruncation": true, "rerankFailurePolicy": "original"}, false)
	require.NoError(t, err)
	require.NotNil(t, opts.RerankTruncation)
	require.True(t, *opts.RerankTruncation)
	require.Equal(t, search.RerankKeepOriginal, opts.RerankFailurePolicy)
}
