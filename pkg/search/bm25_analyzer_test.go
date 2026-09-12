package search

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/blevesearch/snowballstem"
)

func TestBM25AnalyzerConfiguration(t *testing.T) {
	for _, tc := range []struct{ language, canonical string }{
		{"", ""}, {"none", ""}, {" NONE ", ""},
		{"ru", "russian"}, {"Russian", "russian"}, {" RUSSIAN ", "russian"},
		{"en", "english"}, {"English", "english"},
	} {
		t.Run(tc.language, func(t *testing.T) {
			a, err := newBM25Analyzer(tc.language)
			if err != nil {
				t.Fatal(err)
			}
			want := bm25AnalyzerVersion
			if tc.canonical != "" {
				want += "+" + bm25SnowballVersion + ":" + tc.canonical
			}
			if a.identity() != want || !a.compatible(want) {
				t.Fatalf("identity=%q, want %q", a.identity(), want)
			}
			if a.compatible("") {
				t.Fatal("missing analyzer metadata must trigger a rebuild")
			}
			if a.compatible("unknown-analyzer") {
				t.Fatal("accepted unknown analyzer")
			}
		})
	}
	for _, invalid := range []string{"auto", "russain", "ru-RU", "russian;props=*"} {
		if _, err := newBM25Analyzer(invalid); err == nil {
			t.Errorf("accepted %q", invalid)
		}
		if idx, err := NewFulltextIndexWithStemmer(invalid); err == nil || idx != nil {
			t.Errorf("V1 constructor accepted %q", invalid)
		}
		if idx, err := NewFulltextIndexV2WithStemmer(invalid); err == nil || idx != nil {
			t.Errorf("V2 constructor accepted %q", invalid)
		}
	}
}

func TestBM25AnalyzerNormalization(t *testing.T) {
	for _, tc := range []struct {
		name, language, text string
		want                 []string
	}{
		{"neutral", "none", "вагон вагона вагоны Ёлка", []string{"вагон", "вагона", "вагоны", "ёлка"}},
		{"russian", "ru", "ВАГОН, вагона; вагоны", []string{"вагон", "вагон", "вагон"}},
		{"yo folding", "ru", "ЁЛКА е\u0308лки елкой", []string{"елк", "елк", "елк"}},
		{"mixed text", "ru", "CAFÉ 東京 123", []string{"café", "東京", "123"}},
		{"english", "en", "ＲＵＮＮＩＮＧ jumping connections", []string{"run", "jump", "connect"}},
		{"english preserves yo", "en", "ёлка", []string{"ёлка"}},
		{"short words", "ru", "а я и", []string{"а", "я", "и"}},
		{"empty", "ru", "", nil},
		{"punctuation", "ru", " -- !!! \n\t", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, err := newBM25Analyzer(tc.language)
			if err != nil {
				t.Fatal(err)
			}
			if got := a.analyze(tc.text); !slices.Equal(got, tc.want) {
				t.Fatalf("analyze(%q)=%q, want %q", tc.text, got, tc.want)
			}
		})
	}
}

func TestBM25AnalyzerRussianReference(t *testing.T) {
	// These are matched input/output rows from upstream, not expected stems
	// produced by the implementation under test. See testdata/snowball/README.md.
	read := func(name string) []string {
		t.Helper()
		b, err := os.ReadFile(filepath.Join("testdata", "snowball", name))
		if err != nil {
			t.Fatal(err)
		}
		// Git text checkouts may use CRLF on Windows. Only normalize line endings.
		text := strings.ReplaceAll(string(b), "\r\n", "\n")
		return strings.Split(strings.TrimSuffix(text, "\n"), "\n")
	}
	words, stems := read("russian.voc.txt"), read("russian.output.txt")
	if len(words) == 0 || words[0] == "" || len(words) != len(stems) {
		t.Fatalf("fixture alignment: %d inputs, %d outputs", len(words), len(stems))
	}
	a, err := newBM25Analyzer("russian")
	if err != nil {
		t.Fatal(err)
	}
	for i, word := range words {
		if got := a.analyze(word); !slices.Equal(got, []string{stems[i]}) {
			t.Errorf("upstream row %d: %q -> %q, want %q", i+1, word, got, stems[i])
		}
	}
}

func TestBM25AnalyzerStemResultHandling(t *testing.T) {
	a := &bm25Analyzer{stem: func(env *snowballstem.Env) bool {
		env.SetCurrent("")
		return true
	}}
	if got := a.analyze("words"); len(got) != 0 {
		t.Fatalf("must not index empty terms: %q", got)
	}
	a.stem = func(env *snowballstem.Env) bool {
		env.SetCurrent("result")
		return false
	}
	if got := a.analyze("word"); !slices.Equal(got, []string{"result"}) {
		t.Fatalf("discarded valid Current after Stem returned false: %q", got)
	}
}

func TestBM25AnalyzerConcurrentQueries(t *testing.T) {
	a, err := newBM25Analyzer("ru")
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errors := make(chan string, 12)
	for worker := 0; worker < 12; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 250; i++ {
				got := a.analyze("вагоны ЁЛКИ вагон")
				if !slices.Equal(got, []string{"вагон", "елк", "вагон"}) {
					errors <- fmt.Sprintf("concurrent analysis: %q", got)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
}

func TestBM25AnalyzerInvalidEnvironment(t *testing.T) {
	for _, constructor := range []struct {
		name string
		new  func()
	}{
		{"v1", func() { NewFulltextIndex() }},
		{"v2", func() { NewFulltextIndexV2() }},
	} {
		t.Run(constructor.name, func(t *testing.T) {
			t.Setenv(EnvBM25Stemmer, "auto")
			defer func() {
				value := recover()
				if value == nil || !strings.Contains(fmt.Sprint(value), EnvBM25Stemmer) {
					t.Errorf("expected informative configuration panic, got %v", value)
				}
			}()
			constructor.new()
		})
	}
}

func BenchmarkBM25Analyzer(b *testing.B) {
	text := strings.Repeat("Вагоны стоят возле ёлки; running jumping connections. ", 32)
	b.Run("baseline_tokenize", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = tokenize(text)
		}
	})
	for _, language := range []string{"none", "russian", "english"} {
		b.Run(language, func(b *testing.B) {
			a, err := newBM25Analyzer(language)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = a.analyze(text)
			}
		})
	}
}
