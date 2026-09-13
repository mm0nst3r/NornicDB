package nornicgrpc

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/orneryd/nornicdb/pkg/localization"
	gen "github.com/orneryd/nornicdb/pkg/nornicgrpc/gen"
	"github.com/orneryd/nornicdb/pkg/search"
	"golang.org/x/text/language"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
)

// EmbedQueryFunc embeds a query string into a vector.
// Returning (nil, nil) is treated as "embeddings unavailable".
type EmbedQueryFunc func(ctx context.Context, query string) ([]float32, error)

// ChunkQueryFunc splits a query string into embedder-safe chunks.
type ChunkQueryFunc func(ctx context.Context, query string) ([]string, error)

// Searcher is the minimal interface this service needs from the search layer.
type Searcher interface {
	Search(ctx context.Context, query string, embedding []float32, opts *search.SearchOptions) (*search.SearchResponse, error)
}

// Service implements the NornicDB-native gRPC search API.
type Service struct {
	gen.UnimplementedNornicSearchServer

	defaultDatabase string
	maxLimit        int
	rerankEnabled   bool

	embedQuery      EmbedQueryFunc
	chunkQuery      ChunkQueryFunc
	searcher        Searcher
	resolveSearcher func() (Searcher, error)
	localizer       *localization.Manager
}

type Config struct {
	// ResolveSearcher selects the current configured search service for each request.
	// Server bindings use this when configuration can replace the cached service.
	ResolveSearcher func() (Searcher, error)
	DefaultDatabase string
	MaxLimit        int
	// RerankEnabled enables Stage-2 reranking for search when a reranker is configured.
	RerankEnabled bool
	// Localizer renders human-readable status errors. Nil uses en-US.
	Localizer *localization.Manager
}

// NewService creates a NornicDB-native search service.
func NewService(cfg Config, embedQuery EmbedQueryFunc, chunkQuery ChunkQueryFunc, searcher Searcher) (*Service, error) {
	if cfg.Localizer == nil {
		var err error
		cfg.Localizer, err = localization.NewManager(nil, nil)
		if err != nil {
			return nil, status.Error(codes.Internal, localization.GRPCLocalizationInitializationFailed(err).Fallback)
		}
	}
	if searcher == nil {
		service := &Service{localizer: cfg.Localizer}
		return nil, service.localizedStatus(context.Background(), codes.InvalidArgument, localization.SearcherRequired())
	}
	if cfg.MaxLimit <= 0 {
		cfg.MaxLimit = 1000
	}
	if cfg.DefaultDatabase == "" {
		cfg.DefaultDatabase = "nornic"
	}
	return &Service{
		defaultDatabase: cfg.DefaultDatabase,
		maxLimit:        cfg.MaxLimit,
		rerankEnabled:   cfg.RerankEnabled,
		embedQuery:      embedQuery,
		chunkQuery:      chunkQuery,
		searcher:        searcher,
		resolveSearcher: cfg.ResolveSearcher,
		localizer:       cfg.Localizer,
	}, nil
}

func (s *Service) SearchText(ctx context.Context, req *gen.SearchTextRequest) (*gen.SearchTextResponse, error) {
	start := time.Now()

	if req == nil {
		return nil, s.localizedStatus(ctx, codes.InvalidArgument, localization.RequestRequired())
	}
	if req.Query == "" {
		return nil, s.localizedStatus(ctx, codes.InvalidArgument, localization.QueryRequired())
	}
	searcher := s.searcher
	if s.resolveSearcher != nil {
		var err error
		searcher, err = s.resolveSearcher()
		if err != nil {
			return nil, status.Error(codes.Internal, err.Error())
		}
		if searcher == nil {
			return nil, s.localizedStatus(ctx, codes.Internal, localization.SearcherRequired())
		}
	}

	limit := int(req.Limit)
	if limit <= 0 {
		limit = 10
	}
	if limit > s.maxLimit {
		limit = s.maxLimit
	}

	opts := search.DefaultSearchOptions()
	opts.Limit = limit
	opts.RerankEnabled = s.rerankEnabled
	if len(req.Labels) > 0 {
		opts.Types = req.Labels
	}
	if req.MinSimilarity != nil {
		v := float64(*req.MinSimilarity)
		opts.MinSimilarity = &v
	}

	if native, ok := searcher.(interface {
		NativeRerankEnabled() bool
		RerankSearchResponse(context.Context, string, *search.SearchResponse, *search.SearchOptions) error
	}); ok && native.NativeRerankEnabled() {
		opts.RerankEnabled = true
		opts.RerankAfterFusion = native.RerankSearchResponse
	}
	chunkQuery := search.ChunkQueryFunc(nil)
	if s.chunkQuery != nil {
		chunkQuery = func(ctx context.Context, query string) ([]string, error) {
			chunks, err := s.chunkQuery(ctx, query)
			if err != nil {
				return nil, s.localizedStatus(ctx, codes.InvalidArgument, localization.QueryChunkFailed(err))
			}
			return chunks, nil
		}
	}
	resp, err := search.SearchTextChunks(ctx, req.Query, opts, chunkQuery, search.EmbedQueryFunc(s.embedQuery), searcher.Search)
	if err != nil {
		if status.Code(err) != codes.Unknown {
			return nil, err
		}
		return nil, s.localizedStatus(ctx, codes.Internal, localization.SearchFailed(err))
	}

	out := make([]*gen.SearchHit, 0, len(resp.Results))
	for _, r := range resp.Results {
		props, _ := structpb.NewStruct(r.Properties)
		passages := make([]*gen.SupportingPassage, 0, len(r.Passages))
		for _, p := range r.Passages {
			passages = append(passages, &gen.SupportingPassage{NodeId: p.NodeID, ChunkIndex: uint32(p.ChunkIndex), Text: p.Text, MatchedBy: p.MatchedBy, Space: p.Space, SourceFingerprint: p.SourceFingerprint})
		}
		out = append(out, &gen.SearchHit{
			NodeId:     string(r.NodeID),
			Labels:     r.Labels,
			Properties: props,
			Score:      float32(r.Score),
			RrfScore:   float32(r.RRFScore),
			VectorRank: int32(r.VectorRank),
			Bm25Rank:   int32(r.BM25Rank),
			Passages:   passages,
		})
	}

	if resp.Rerank != nil {
		report, _ := json.Marshal(resp.Rerank)
		grpc.SetTrailer(ctx, metadata.Pairs("nornicdb-rerank", string(report)))
	}
	return &gen.SearchTextResponse{
		SearchMethod:      resp.SearchMethod,
		Hits:              out,
		FallbackTriggered: resp.FallbackTriggered,
		Message:           resp.Message,
		TimeSeconds:       time.Since(start).Seconds(),
	}, nil
}

func (s *Service) localizedStatus(ctx context.Context, code codes.Code, message localization.Message) error {
	preferences := []language.Tag(nil)
	if incoming, ok := metadata.FromIncomingContext(ctx); ok {
		values := incoming.Get("accept-language")
		if len(values) > 0 {
			preferences, _, _ = language.ParseAcceptLanguage(strings.Join(values, ","))
		}
	}
	if len(preferences) > 0 {
		match := s.localizer.Resolve("grpc", preferences...)
		ctx = localization.WithPreferences(ctx, match.Tag)
	}
	text, _, err := s.localizer.Render(ctx, message)
	if err != nil {
		text = message.Fallback
	}
	return status.Error(code, text)
}
