package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/orneryd/nornicdb/pkg/nornicdb"
	"github.com/orneryd/nornicdb/pkg/storage"
	"github.com/stretchr/testify/require"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Public node deletion must be visible to a warmed HTTP count before flush and after reopen.
func TestHTTPCountReflectsPublicNodeDeletion(t *testing.T) {
	for _, queuedUpdate := range []bool{false, true} {
		t.Run(fmt.Sprintf("queued_update_%v", queuedUpdate), func(t *testing.T) {
			dir := t.TempDir()
			cfg := nornicdb.DefaultConfig()
			cfg.Memory.DecayEnabled = false
			cfg.Memory.AutoLinksEnabled = false
			cfg.Database.AsyncWritesEnabled = true
			db, err := nornicdb.Open(dir, cfg)
			require.NoError(t, err)
			defer func() {
				if db != nil {
					require.NoError(t, db.Close())
				}
			}()
			ae, ok := db.GetBaseStorageForManager().(*storage.AsyncEngine)
			require.True(t, ok)
			sc := DefaultConfig()
			sc.EmbeddingEnabled = false
			sc.Port = 0
			app, err := New(db, nil, sc)
			require.NoError(t, err)
			hs := httptest.NewServer(app.buildRouter())
			defer hs.Close()
			defer app.Stop(context.Background())
			query := func(statement string) float64 {
				t.Helper()
				body, err := json.Marshal(map[string]any{"statements": []any{map[string]any{"statement": statement}}})
				require.NoError(t, err)
				resp, err := hs.Client().Post(hs.URL+"/db/nornic/tx/commit", "application/json", bytes.NewReader(body))
				require.NoError(t, err)
				defer resp.Body.Close()
				require.Equal(t, http.StatusOK, resp.StatusCode)
				var result struct {
					Errors  []any `json:"errors"`
					Results []struct {
						Data []struct {
							Row []any `json:"row"`
						} `json:"data"`
					} `json:"results"`
				}
				require.NoError(t, json.NewDecoder(resp.Body).Decode(&result))
				require.Empty(t, result.Errors)
				require.Len(t, result.Results, 1)
				require.Len(t, result.Results[0].Data, 1)
				return result.Results[0].Data[0].Row[0].(float64)
			}
			const warmed = "MATCH (n:AsyncDeleteProof) RETURN count(n)"
			id := storage.NodeID("http-count-causal")
			_, err = db.GetStorage().CreateNode(&storage.Node{ID: id, Labels: []string{"AsyncDeleteProof"}, Properties: map[string]any{"content": "original"}})
			require.NoError(t, err)
			require.NoError(t, ae.Flush())
			require.Equal(t, float64(1), query(warmed))
			release := ae.HoldFlush()
			func() {
				defer release()
				if queuedUpdate {
					node, err := db.GetStorage().GetNode(id)
					require.NoError(t, err)
					node.Properties["content"] = "edited"
					require.NoError(t, db.GetStorage().UpdateNode(node))
				}
				require.NoError(t, db.DeleteNode(context.Background(), string(id)))
				_, err = db.GetStorage().GetNode(id)
				require.ErrorIs(t, err, storage.ErrNotFound)
				count, err := db.GetStorage().(storage.LabelStatsEngine).NodeCountByLabel("AsyncDeleteProof")
				require.NoError(t, err)
				require.Zero(t, count)
				warm := query(warmed)
				require.Zero(t, warm)
				cold := query("MATCH (fresh:AsyncDeleteProof) RETURN count(fresh)")
				require.Zero(t, cold)
				t.Logf("before flush: storage GetNode=ErrNotFound storage label count=%d warmed HTTP=%v cold HTTP=%v", count, warm, cold)
			}()
			require.NoError(t, ae.Flush())
			require.Zero(t, query(warmed))
			hs.Close()
			require.NoError(t, app.Stop(context.Background()))
			require.NoError(t, db.Close())
			db = nil
			reopened, err := nornicdb.Open(dir, cfg)
			require.NoError(t, err)
			defer reopened.Close()
			_, err = reopened.GetStorage().GetNode(id)
			require.ErrorIs(t, err, storage.ErrNotFound)
			app2, err := New(reopened, nil, sc)
			require.NoError(t, err)
			defer app2.Stop(context.Background())
			hs = httptest.NewServer(app2.buildRouter())
			defer hs.Close()
			require.Zero(t, query(warmed))
			t.Log("after reopen: GetNode=ErrNotFound warmed query HTTP=0")
		})
	}
}
