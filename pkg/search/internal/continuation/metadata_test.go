package continuation

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestRetainedMetadataUsesExistingMemoryBudgets(t *testing.T) {
	config := DefaultConfig()
	config.MaxBuildBytes = 1024
	if _, err := NewBuilder(RankedOnly, false, []Hit{{ID: "a", Metadata: strings.Repeat("x", 2048)}}, config); !errors.Is(err, ErrCapacity) {
		t.Fatalf("oversized hit metadata must respect build budget: %v", err)
	}
	p := Population{Mode: RankedOnly, Metadata: strings.Repeat("x", 2048)}
	if _, _, err := clonePopulation(context.Background(), p, config); !errors.Is(err, ErrCapacity) {
		t.Fatalf("oversized report metadata must respect retained budget: %v", err)
	}
}
