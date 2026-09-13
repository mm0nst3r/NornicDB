package config

import (
	"github.com/stretchr/testify/require"
	"os"
	"path/filepath"
	"testing"
)

func TestVoyageRerankYAMLAndEnvironment(t *testing.T) {
	clearEnvVars(t)
	path := filepath.Join(t.TempDir(), "voyage.yaml")
	require.NoError(t, os.WriteFile(path, []byte("search_rerank:\n  enabled: true\n  provider: voyage\n  model: rerank-2.5\n  truncation: true\n  failure_policy: original\n"), 0600))
	cfg, err := LoadFromFile(path)
	require.NoError(t, err)
	require.Equal(t, "voyage", cfg.Features.SearchRerankProvider)
	require.True(t, cfg.Features.SearchRerankTruncation)
	require.Equal(t, "original", cfg.Features.SearchRerankFailurePolicy)
	t.Setenv("NORNICDB_SEARCH_RERANK_TRUNCATION", "false")
	t.Setenv("NORNICDB_SEARCH_RERANK_FAILURE_POLICY", "error")
	cfg = LoadFromEnv()
	require.False(t, cfg.Features.SearchRerankTruncation)
	require.Equal(t, "error", cfg.Features.SearchRerankFailurePolicy)
}
