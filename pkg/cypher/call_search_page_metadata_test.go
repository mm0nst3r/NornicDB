package cypher

import (
	"encoding/json"
	"testing"

	"github.com/orneryd/nornicdb/pkg/search"
	"github.com/stretchr/testify/require"
)

func TestNativeSearchPagePreservesMetadataAndCoreTypes(t *testing.T) {
	p := &search.SearchPageResponse{Total: 1, Results: []search.SearchPageHit{{ID: "original", Metadata: json.RawMessage(`{"passages":[{"text":"original full passage\nline two","index":9007199254740993}],"score":99,"group_key":"bad"}`)}},
		Metadata: json.RawMessage(`{"rerank":{"status":"applied","candidates":101,"elapsed":1.5},"total":999,"eligible_count":999}`)}
	native, err := nativeSearchPage(p)
	require.NoError(t, err)
	require.Equal(t, int64(1), native["total"])
	require.Nil(t, native["eligible_count"])
	require.Equal(t, "", native["next_cursor"])
	r := native["rerank"].(map[string]interface{})
	require.Equal(t, int64(101), r["candidates"])
	require.Equal(t, 1.5, r["elapsed"])
	hit := native["results"].([]interface{})[0].(map[string]interface{})
	require.Equal(t, float64(0), hit["score"])
	require.Equal(t, "", hit["group_key"])
	passage := hit["passages"].([]interface{})[0].(map[string]interface{})
	require.Equal(t, int64(9007199254740993), passage["index"])
	require.Equal(t, "original full passage\nline two", passage["text"])
	r["status"] = "changed"
	passage["text"] = "changed"
	again, err := nativeSearchPage(p)
	require.NoError(t, err)
	require.Equal(t, "applied", again["rerank"].(map[string]interface{})["status"])
	_, err = nativeSearchPage(&search.SearchPageResponse{Metadata: []byte(`bad`)})
	require.Error(t, err)
}
