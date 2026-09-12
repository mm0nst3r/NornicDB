package nornicgrpc

import (
	"context"
	"encoding/json"
	"sort"
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

	native, _ := searcher.(interface {
		NativeRerankEnabled() bool
		RerankSearchResponse(context.Context, string, *search.SearchResponse, *search.SearchOptions) error
	})
	if native != nil && native.NativeRerankEnabled() {
		opts.RerankEnabled = true
	}
	nativeAfterFusion := false
	// If embeddings are available, proactively chunk long queries by length.
	// This keeps vector search usable for paragraph-sized inputs without relying
	// on embedder/tokenizer failures to detect "too long" queries.
	const (
		queryChunkSize    = 512
		queryChunkOverlap = 50
		maxQueryChunks    = 32
		outerRRFK         = 60
	)

	var (
		resp *search.SearchResponse
		err  error
	)

	if s.embedQuery != nil {
		queryChunks := []string{req.Query}
		if s.chunkQuery != nil {
			queryChunks, err = s.chunkQuery(ctx, req.Query)
			if err != nil {
				return nil, s.localizedStatus(ctx, codes.InvalidArgument, localization.QueryChunkFailed(err))
			}
		}
		if len(queryChunks) > maxQueryChunks {
			queryChunks = queryChunks[:maxQueryChunks]
		}

		if len(queryChunks) <= 1 {
			emb, embedErr := s.embedQuery(ctx, req.Query)
			if embedErr == nil && len(emb) > 0 {
				resp, err = searcher.Search(ctx, req.Query, emb, opts)
			}
		} else {
			// Pull more candidates per chunk, then cut down after fusion.
			perChunkLimit := limit
			if perChunkLimit < 10 {
				perChunkLimit = 10
			}
			if perChunkLimit < limit*3 {
				perChunkLimit = limit * 3
			}
			if perChunkLimit > 100 {
				perChunkLimit = 100
			}

			type fused struct {
				best     search.SearchResult
				hasBest  bool
				scoreRRF float64
			}
			fusedByID := make(map[string]*fused)

			nativeAfterFusion = opts.RerankEnabled && native != nil && native.NativeRerankEnabled()
			if nativeAfterFusion {
				perChunkLimit = opts.RerankTopK
			}
			var usedVectorChunks int
			for _, chunkQuery := range queryChunks {
				emb, embedErr := s.embedQuery(ctx, chunkQuery)
				if embedErr != nil || len(emb) == 0 {
					continue
				}
				usedVectorChunks++

				chunkOpts := *opts
				chunkOpts.Limit = perChunkLimit
				if nativeAfterFusion {
					chunkOpts.RerankEnabled = false
				}
				chunkResp, searchErr := searcher.Search(ctx, chunkQuery, emb, &chunkOpts)
				if searchErr != nil || chunkResp == nil {
					continue
				}

				for rank := range chunkResp.Results {
					r := chunkResp.Results[rank]
					id := string(r.NodeID)
					f := fusedByID[id]
					if f == nil {
						f = &fused{}
						fusedByID[id] = f
					}
					// Outer RRF: 1/(k + rank), rank is 1-based.
					f.scoreRRF += 1.0 / (outerRRFK + float64(rank+1))
					if !f.hasBest || r.Score > f.best.Score {
						f.best = r
						f.hasBest = true
					}
				}
			}

			if usedVectorChunks > 0 && len(fusedByID) > 0 {
				fusedList := make([]*fused, 0, len(fusedByID))
				for _, f := range fusedByID {
					fusedList = append(fusedList, f)
				}
				sort.Slice(fusedList, func(i, j int) bool {
					return fusedList[i].scoreRRF > fusedList[j].scoreRRF
				})
				if !nativeAfterFusion && len(fusedList) > limit {
					fusedList = fusedList[:limit]
				}

				resp = &search.SearchResponse{
					SearchMethod:      "chunked_rrf_hybrid",
					FallbackTriggered: false,
					Results:           make([]search.SearchResult, 0, len(fusedList)),
				}
				for _, f := range fusedList {
					r := f.best
					r.Score = f.scoreRRF
					r.RRFScore = f.scoreRRF
					r.VectorRank = 0
					r.BM25Rank = 0
					resp.Results = append(resp.Results, r)
				}
			}
		}
	}

	if nativeAfterFusion && resp != nil {
		if err := native.RerankSearchResponse(ctx, req.Query, resp, opts); err != nil {
			return nil, s.localizedStatus(ctx, codes.Internal, localization.SearchFailed(err))
		}
	}
	// If vector path didn't produce a response, fall back to BM25.
	if resp == nil {
		resp, err = searcher.Search(ctx, req.Query, nil, opts)
		if err != nil {
			return nil, s.localizedStatus(ctx, codes.Internal, localization.SearchFailed(err))
		}
	} else if err != nil {
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
