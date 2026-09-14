package voyage

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"
)

func TestErrorRenderingRetainsKnownProviderDiagnostics(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusOK} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			client := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("X-Request-ID", "req-failed-123")
				w.WriteHeader(status)
				// A successful HTTP envelope with an invalid document index has
				// useful usage metadata, while neither failure may echo its body.
				_, _ = fmt.Fprintf(w, `{"data":[{"index":99,"relevance_score":0.7}],"model":"rerank-2.5","usage":{"total_tokens":123},"error":"%s private submitted content"}`, testKey)
			})
			_, err := client.Rerank(context.Background(), rerankRequest())
			kind := Authentication
			if status == http.StatusOK {
				kind = MalformedResponse
			}
			providerError := requireKind(t, err, kind)
			if providerError.Metadata.RequestID != "req-failed-123" {
				t.Fatal("fixture did not establish available provider identity")
			}
			text := err.Error()
			for _, required := range []string{kind, "req-failed-123", `"attempts":1`} {
				if !strings.Contains(text, required) {
					t.Fatalf("public failure rendering lost %q: %s", required, text)
				}
			}
			if status == http.StatusOK && !strings.Contains(text, `"total_tokens":123`) {
				t.Fatalf("public failure rendering lost available usage: %s", text)
			}
			for _, forbidden := range []string{testKey, "private submitted content", client.baseURL} {
				if strings.Contains(text, forbidden) {
					t.Fatal("public failure exposed submitted content, credential or URL")
				}
			}
		})
	}
}
