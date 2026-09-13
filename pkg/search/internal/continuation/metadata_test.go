package continuation

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMetadataGroupedPagesOwnCopiesAndPreserveWireFields(t *testing.T) {
	ctx := context.Background()
	original := json.RawMessage(`{"passages":[{"text":"complete supporting passage\nsecond line","index":9007199254740993}],"id":"cannot-override","group_key":"cannot-override"}`)
	input := []Hit{{ID: "a-low", Score: 1, Metadata: json.RawMessage(`{"passages":["lower-ranked"]}`)},
		{ID: "a-best", Score: 3, Metadata: bytes.Clone(original)}, {ID: "b-best", Score: 2, Metadata: bytes.Clone(original)}}
	b, err := NewBuilder(RankedThenID, true, input, DefaultConfig())
	require.NoError(t, err)
	input[1].Metadata[0] = '!'
	for _, pair := range [][2]string{{"a-low", "A"}, {"a-best", "A"}, {"b-best", "B"}, {"c-tail", "C"}} {
		require.NoError(t, b.Add(pair[0], pair[1]))
	}
	p, err := b.Finish(ctx)
	require.NoError(t, err)
	require.Equal(t, "a-best", p.Hits[0].ID)
	require.Equal(t, original, p.Hits[0].Metadata)
	p.Metadata = json.RawMessage(`{"rerank":{"status":"applied","candidates":3},"total":999,"next_cursor":"cannot-override"}`)
	wantReport := json.RawMessage(bytes.Clone(p.Metadata))
	var store Store
	ticket, err := store.Start(ctx)
	require.NoError(t, err)
	first, err := ticket.Commit(ctx, [32]byte{}, p, 1)
	require.NoError(t, err)
	p.Hits[1].Metadata[0] = '!'
	p.Metadata[0] = '!'
	first.Metadata[0] = '!'
	first.Results[0].Metadata[0] = '!'
	second, err := store.Continue(ctx, first.NextCursor, [32]byte{}, 1)
	require.NoError(t, err)
	require.Equal(t, wantReport, second.Metadata)
	require.Equal(t, original, second.Results[0].Metadata)
	encoded, err := json.Marshal(second)
	require.NoError(t, err)
	var wire map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(encoded, &wire))
	require.JSONEq(t, `{"status":"applied","candidates":3}`, string(wire["rerank"]))
	require.JSONEq(t, `3`, string(wire["total"]))
	require.NotContains(t, wire, "metadata")
	var hits []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(wire["results"], &hits))
	require.JSONEq(t, `"b-best"`, string(hits[0]["id"]))
	require.Contains(t, string(hits[0]["passages"]), "9007199254740993")
	second.Metadata[0] = '!'
	second.Results[0].Metadata[0] = '!'
	replay, err := store.Continue(ctx, first.NextCursor, [32]byte{}, 1)
	require.NoError(t, err)
	replayed, err := json.Marshal(replay)
	require.NoError(t, err)
	require.JSONEq(t, string(encoded), string(replayed))
	last, err := store.Continue(ctx, replay.NextCursor, [32]byte{}, 1)
	require.NoError(t, err)
	require.Equal(t, wantReport, last.Metadata)
	require.Empty(t, last.Results[0].Metadata)
	lastJSON, err := json.Marshal(last)
	require.NoError(t, err)
	require.NotContains(t, string(lastJSON), "cannot-override")
}

func TestMetadataLimitsIncludeHitAndPopulationPayloads(t *testing.T) {
	raw := json.RawMessage(`{"diagnostic":"` + strings.Repeat("x", 1000) + `"}`)
	_, err := NewBuilder(RankedOnly, false, []Hit{{ID: "a", Metadata: raw}}, Config{MaxBuildBytes: 1000})
	require.ErrorIs(t, err, ErrCapacity)
	for _, onHit := range []bool{false, true} {
		for _, buildLimit := range []bool{false, true} {
			config := DefaultConfig()
			if buildLimit {
				config.MaxBuildBytes = 1000
			} else {
				config.MaxBytes = 1000
			}
			p := Population{Mode: IDOnly, Hits: []Hit{{ID: "a", Phase: CatalogPhase}}}
			if onHit {
				p.Hits[0].Metadata = raw
			} else {
				p.Metadata = raw
			}
			_, _, err := clonePopulation(context.Background(), p, config)
			require.ErrorIs(t, err, ErrCapacity)
		}
	}
}

func TestMetadataRejectsInvalidObjectsAndCannotReplaceOmittedCoreFields(t *testing.T) {
	for _, invalid := range []json.RawMessage{[]byte(`[]`), []byte(`null`), []byte(`{"broken"`)} {
		_, err := NewBuilder(RankedOnly, false, []Hit{{ID: "a", Metadata: invalid}}, DefaultConfig())
		require.ErrorIs(t, err, ErrInvalidRequest)
		_, _, err = clonePopulation(context.Background(), Population{Mode: IDOnly, Metadata: invalid}, DefaultConfig())
		require.ErrorIs(t, err, ErrInvalidRequest)
		_, _, err = clonePopulation(context.Background(), Population{Mode: IDOnly, Hits: []Hit{{ID: "a", Phase: CatalogPhase, Metadata: invalid}}}, DefaultConfig())
		require.ErrorIs(t, err, ErrInvalidRequest)
		_, err = json.Marshal(Hit{Metadata: invalid})
		require.Error(t, err)
	}
	encoded, err := json.Marshal(Hit{ID: "a", Metadata: []byte(`{"similarity":123,"passages":["whole"]}`)})
	require.NoError(t, err)
	require.NotContains(t, string(encoded), "similarity")
	require.Contains(t, string(encoded), `"passages"`)
}
