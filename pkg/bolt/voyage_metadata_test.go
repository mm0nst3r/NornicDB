package bolt

import (
	"github.com/stretchr/testify/require"
	"net"
	"testing"
)

func TestVoyageMetadataSurvivesEmptyPullAndDiscard(t *testing.T) {
	for _, discard := range []bool{false, true} {
		t.Run(map[bool]string{false: "pull", true: "discard"}[discard], func(t *testing.T) {
			client, server := net.Pipe()
			defer client.Close()
			defer server.Close()
			session := newTestSession(server, &mockExecutor{})
			session.lastResult = &QueryResult{Columns: []string{"node"}, Rows: [][]any{}, Metadata: map[string]any{"rerank": map[string]any{"status": "applied", "returned": int64(0)}}}
			received := make(chan map[string]any, 1)
			go func() {
				_, data, err := ReadMessage(client)
				if err != nil {
					received <- nil
					return
				}
				meta, _, err := decodePackStreamMap(data, 0)
				if err != nil {
					received <- nil
					return
				}
				received <- meta
			}()
			var err error
			if discard {
				err = session.handleDiscard(nil)
			} else {
				err = session.handlePull(nil)
			}
			require.NoError(t, err)
			metadata := <-received
			require.NotNil(t, metadata)
			report, ok := metadata["rerank"].(map[string]any)
			require.True(t, ok)
			require.Equal(t, "applied", report["status"])
			require.EqualValues(t, 0, report["returned"])
		})
	}
}
