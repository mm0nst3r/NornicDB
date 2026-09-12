# Optional Snowball stemming for BM25

BM25 normally uses language-neutral Unicode NFKC normalization, case folding,
and tokenization. Stemming is an optional, explicitly selected step after that
existing analysis. The same immutable analyzer is used for indexing, bulk
indexing, updates, deletion, and queries in both BM25 engines.

## Configuration

Set `NORNICDB_BM25_STEMMER` before constructing the search service or starting
NornicDB:

```sh
export NORNICDB_BM25_STEMMER=russian
```

PowerShell:

```powershell
$env:NORNICDB_BM25_STEMMER = "russian"
```

Supported values are `none` (also an empty/unset value), `russian`/`ru`, and
`english`/`en`. Values are case-insensitive and surrounding whitespace is
ignored. This initial integration exposes Russian and English, not every
language supported by Snowball. It does not select a language automatically.
The configured stemmer is applied to every token in that index.

An invalid setting is not silently ignored. The existing no-error-return
constructors panic before creating an index, with the environment variable and
invalid value in the error. Applications that require ordinary error handling
can use the explicit constructors instead:

```go
index, err := search.NewFulltextIndexV2WithStemmer("russian")
if err != nil {
    return err
}
index.Index("transcript-1", "вагон")
```

`NewFulltextIndexWithStemmer` provides the equivalent V1 API. These explicit
constructors do not read `NORNICDB_BM25_STEMMER`; V2's existing prefix-matching
configuration still applies. `AnalyzerIdentity()` identifies the configuration
captured by the index. Changing the process environment does not mutate an
existing index. To change its language, recreate the service/index and rebuild.

### Using a separate NornicDB server

Set `NORNICDB_BM25_STEMMER=russian` in the server's environment, for example with
`-e NORNICDB_BM25_STEMMER=russian` when starting its Docker container. Send the
original text through your normal Bolt or HTTP Cypher client:

```cypher
CREATE (:Transcript {id: 'wagon-1', content: 'вагон'});
CALL db.retrieve({query: 'вагонами', limit: 10}) YIELD node, score
RETURN node.id, node.content, score;
```

With lexical retrieval, the result retains `content: 'вагон'`. No application-side
stemmed/normalized copy is needed. The existing server setting
`NORNICDB_SEARCH_BM25_PROPERTIES=title,content` limits indexing to those original
property values; choose the fields your application actually stores. Without an
allowlist, NornicDB indexes all searchable properties, so also storing a normalized
copy would index that extra field. The allowlist participates in build settings.

Both settings are process-wide and apply to search services for each logical
database in that server. They do not provide different languages per database or
field. Stemming affects lexical retrieval only; stored vectors, embedding
generation and vector scoring are independent of this setting.

## Search behavior

Russian `вагон`, `вагона`, and `вагоны` produce the same lexical term. Russian
analysis also folds `ё` to `е`, so forms such as `Ёлка` and `елки` can match.
English `running` and `runs` produce `run`.

Stemming only changes lexical terms. Original document text, display text,
and the text supplied to downstream reranking remain unchanged. Literal
`PhraseSearch` retains its existing behavior: it does not become a stemmed
phrase query. Existing prefix-matching policies remain in effect, now operating
on analyzed terms. No new stopword policy, automatic language detection,
embedding transformation, or storage-schema change is introduced.

This is BM25 search-package configuration. It does not add a Cypher analyzer
DDL option or claim to replace every independent full-text query implementation
in the repository. Language-neutral tokenization used outside the two BM25
indexes is left unchanged.

## Persistence and rebuilding

The analyzer identity includes the Unicode analyzer revision, the pinned
Snowball implementation, the adapter revision, and the canonical language.
It is stored in both BM25 snapshot formats and the service build-settings
fingerprint. The fingerprint comes from the actual index, not a later read of
the environment.

Both `Save` and `SaveNoCopy` record the identity. `Load` rejects an incompatible
analyzer and clears the index so the existing caller rebuild mechanism can run.
A standalone API caller must likewise repopulate an index after a rejected
snapshot; `Load` does not fetch the source documents on the caller's behalf.

Snapshot format versions advance from V1 `1.1.0` to `1.2.0` and V2 `2.1.0` to
`2.2.0`. Existing `1.1.0`/`2.1.0` snapshots therefore rebuild even when stemming
is disabled. This also makes older binaries reject the new formats rather than
unknowingly searching stemmed postings with neutral query analysis. It does not
alter the graph data or the WAL.

Current V1 snapshots can migrate into V2 only with a matching analyzer identity.
Older snapshots, including legacy `1.0.0`, and snapshots without analyzer
metadata are rejected for rebuilding. Updating
the dependency or adapter requires reviewing whether to advance the analyzer
revision and rebuild affected indexes.

## Implementation provenance

The integration pins `github.com/blevesearch/snowballstem v0.9.0`, which contains
Go code generated by Snowball. It does **not** claim to contain newly generated
Snowball 3.1.1 algorithms. This version's Russian generated code predates the
upstream `ё`-to-`е` prelude; the small Russian-specific adapter applies that
upstream rule before invoking the generated algorithm. It is not applied to
neutral or English analysis. The adapter revision is part of persistence
compatibility.

Snowball environments are mutable, so each analysis call uses its own
execution environment, reusing it across that call's tokens only. No mutable
Snowball environment is shared by simultaneous readers.

Sources:

- https://github.com/blevesearch/snowballstem/tree/v0.9.0
- https://snowballstem.org/algorithms/russian/stemmer.html
- https://github.com/snowballstem/snowball/tree/main/go

The dependency's copyright and redistribution terms are reproduced in
[the Snowball notice](../third-party/snowballstem-COPYING.txt).

## Tests and benchmarks

The change supplies analyzer, mutation, persistence, migration, configuration,
concurrent-analyzer, and service rebuild tests. The service tests verify that
changing languages rebuilds BM25 from graph storage and updates persisted build
settings, while reopening with the same language reuses the BM25 snapshot.
The included Russian vocabulary/output fixture
is a **96-pair subset** of the upstream reference corpus, not the full corpus.
See [fixture provenance](../../pkg/search/testdata/snowball/README.md).

From a checkout with the repository's required Go toolchain:

```sh
go test ./pkg/search -run 'TestBM25(Analyzer|Stemming)' -count=1
go test -race ./pkg/search -run 'TestBM25(Analyzer|Stemming)' -count=1
go test ./pkg/search -count=1
go test ./pkg/search -run '^$' -bench '^BenchmarkBM25Analyzer$' -benchmem
```

The benchmark includes the existing tokenizer as a neutral baseline and the
neutral, Russian, and English analyzers.
