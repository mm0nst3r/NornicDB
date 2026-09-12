package search

import (
	"fmt"
	"os"
	"strings"

	"github.com/blevesearch/snowballstem"
	"github.com/blevesearch/snowballstem/english"
	"github.com/blevesearch/snowballstem/russian"
)

// EnvBM25Stemmer selects the optional BM25 Snowball stemmer. Supported values are
// "none" (the default), "russian"/"ru", and "english"/"en". The setting is captured
// when an index is constructed; changing it requires recreating the index and
// rebuilding incompatible persisted data. It does not affect stored text,
// embeddings, or literal phrase matching.
const EnvBM25Stemmer = "NORNICDB_BM25_STEMMER"

// Include both the dependency and adapter revisions: changes to either may
// change index terms. The Russian adapter supplies Snowball's 2018 ё -> е prelude,
// which is absent from the Go code generated in snowballstem v0.9.0.
const bm25SnowballVersion = "snowballstem-v0.9.0-adapter-v1"

// bm25Analyzer is immutable. Snowball's mutable Env is deliberately not retained
// here, because multiple searches may analyze their queries concurrently.
// A nil analyzer is the existing language-neutral analyzer.
type bm25Analyzer struct {
	language string
	stem     func(*snowballstem.Env) bool
}

func newBM25Analyzer(language string) (*bm25Analyzer, error) {
	switch strings.ToLower(strings.TrimSpace(language)) {
	case "", "none":
		return nil, nil
	case "russian", "ru":
		return &bm25Analyzer{language: "russian", stem: russian.Stem}, nil
	case "english", "en":
		return &bm25Analyzer{language: "english", stem: english.Stem}, nil
	default:
		return nil, fmt.Errorf("unsupported BM25 stemmer %q: use none, russian (ru), or english (en)", language)
	}
}

func mustBM25AnalyzerFromEnv() *bm25Analyzer {
	analyzer, err := newBM25Analyzer(os.Getenv(EnvBM25Stemmer))
	if err != nil {
		// Existing index constructors cannot return errors. Fail before creating
		// an index rather than silently indexing with the wrong analyzer. Callers
		// needing error handling can use the WithStemmer constructors instead.
		panic(fmt.Errorf("%s: %w", EnvBM25Stemmer, err))
	}
	return analyzer
}

func (a *bm25Analyzer) identity() string {
	if a == nil {
		return bm25AnalyzerVersion
	}
	return bm25AnalyzerVersion + "+" + bm25SnowballVersion + ":" + a.language
}

func (a *bm25Analyzer) compatible(saved string) bool {
	return saved == a.identity()
}

func (a *bm25Analyzer) analyze(text string) []string {
	tokens := tokenize(text)
	if a == nil || len(tokens) == 0 {
		return tokens
	}

	// One environment per analysis call, reused across its tokens, never shared
	// by concurrent readers. Normalization and splitting remain in tokenize.
	env := snowballstem.NewEnv("")
	out := tokens[:0]
	for _, token := range tokens {
		if a.language == "russian" {
			// This is the upstream Russian algorithm's prelude, not global Unicode
			// normalization: neutral and English analysis must preserve ё.
			token = strings.ReplaceAll(token, "ё", "е")
		}
		env.SetCurrent(token)
		// Snowball's bool describes control flow, not whether Current is valid
		// or whether it changed. Use Current even when Stem returns false.
		a.stem(env)
		if stem := env.Current(); stem != "" {
			out = append(out, stem)
		}
	}
	return out
}

// NewFulltextIndexWithStemmer creates a V1 index using an explicit language,
// independent of EnvBM25Stemmer. Invalid languages return an error without
// constructing an index. An empty language or "none" preserves neutral analysis.
//
// Example:
//
//	index, err := search.NewFulltextIndexWithStemmer("russian")
//	if err != nil {
//		return err
//	}
//	index.Index("transcript-1", "вагон")
func NewFulltextIndexWithStemmer(language string) (*FulltextIndex, error) {
	analyzer, err := newBM25Analyzer(language)
	if err != nil {
		return nil, err
	}
	return newFulltextIndexWithAnalyzer(analyzer), nil
}

// NewFulltextIndexV2WithStemmer creates a V2 index using an explicit language,
// independent of EnvBM25Stemmer. Its language and error semantics are identical
// to NewFulltextIndexWithStemmer. Existing V2 prefix settings still apply.
//
// Example:
//
//	index, err := search.NewFulltextIndexV2WithStemmer("ru")
//	if err != nil {
//		return err
//	}
//	index.IndexBatch([]search.FulltextBatchEntry{{ID: "transcript-1", Text: "вагон"}})
func NewFulltextIndexV2WithStemmer(language string) (*FulltextIndexV2, error) {
	analyzer, err := newBM25Analyzer(language)
	if err != nil {
		return nil, err
	}
	return newFulltextIndexV2WithAnalyzer(analyzer), nil
}

// AnalyzerIdentity returns the immutable lexical analyzer identity used by this
// V1 index. Saved-index compatibility includes this identity.
func (f *FulltextIndex) AnalyzerIdentity() string { return f.analyzer.identity() }

// AnalyzerIdentity returns the immutable lexical analyzer identity used by this
// V2 index. Saved-index compatibility includes this identity.
func (f *FulltextIndexV2) AnalyzerIdentity() string { return f.analyzer.identity() }

func (s *Service) bm25AnalyzerIdentity() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index, ok := s.fulltextIndex.(interface{ AnalyzerIdentity() string }); ok {
		return index.AnalyzerIdentity()
	}
	// Disabled indexes and existing test implementations need no new method
	// on the bm25Index interface.
	return bm25AnalyzerVersion
}
