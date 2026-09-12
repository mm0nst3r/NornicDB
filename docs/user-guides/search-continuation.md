# Native search continuation

`Service.SearchPage` adds process-local continuation without changing the
existing `Service.Search` API. It runs one initial retrieval/materialisation and
then returns slices of a fixed ID/score population. It does **not** increase ANN
recall or produce a global relevance ranking. The same implementation is exposed
by `db.retrieve.page` and `db.retrieve.release` through native Cypher, including
Bolt drivers and the existing HTTP Cypher endpoint.

## Native clients

Send an empty cursor initially, then repeat the request with `page.next_cursor`:

```cypher
CALL db.retrieve.page({
  query: $query, embedding: $embedding,
  filters: {collection: ['summer']},
  mode: 'ranked_then_id', limit: 200, pageSize: 50,
  groupBy: 'asset_id', cursor: $cursor
}) YIELD page
RETURN page
```

The procedure returns one `page` map even when there are no hits. Its `results`
list contains descriptors; its count, exhaustion and cursor fields describe the
whole page. Stop when `next_cursor` is empty. To release a cursor early, send the
same request map to `CALL db.retrieve.release($request) YIELD released`.

Vectors are supplied explicitly and can come from any provider or supported
dimension. To use the database's configured embedder, call
`CALL db.index.vector.embed($query) YIELD embedding` once and retain that vector
with the query for subsequent pages. A text query without a vector uses BM25;
an empty text query with a vector uses vector retrieval. Metadata-only browsing
uses `mode: 'id'` and needs neither. Pagination makes no embedding-provider calls.
Set `rerank: true` to use a configured reranker where the existing retrieval path
supports it. Only the initial ranked pool is reranked; later pages and the ID tail
do not call the reranker again.

The ordinary retrieval option parser is shared with `db.retrieve`; page requests
also accept `page_size` and `group_by`. The server derives cursor scope from the
authenticated principal and roles, and checks current query/database permissions
on every call. Query arguments cannot supply an authenticated scope. Embedded
executors using permission-checker contexts must also supply a trusted scope with
`cypher.WithSearchContinuationScope`.

State errors have distinct native codes, for example
`Neo.ClientError.Statement.SearchCursorInvalidated`, `SearchCursorExpired`,
`SearchCursorMismatch`, and `SearchCursorInvalid` under the same prefix. These
require explicit client handling, not automatic transaction retries.

Dedicated REST search, MCP discover, GraphQL search, native gRPC search and Qdrant
compatibility retain their existing contracts. This change supplies a common Go
and native Cypher continuation interface; it does not claim those distinct search
response formats now contain cursors. In particular, query chunk aggregation in
REST/MCP/gRPC and Qdrant scrolling are different populations.

## Choose the population explicitly

| Mode | Population | Order | Complete collection enumeration? |
|---|---|---|---|
| `SearchPageRanked` (`ranked`, default) | The bounded results selected by one search pipeline invocation | Descending `SearchResult.Score`, then ascending logical ID | No |
| `SearchPageRankedThenID` (`ranked_then_id`) | Ranked prefix plus every other metadata-eligible member | Ranked prefix, then unscored logical-ID order | Yes, within the configured materialisation limits and mutation contract |
| `SearchPageID` (`id`) | Every metadata-eligible member; no retrieval/index warming | Ascending logical ID | Yes, within the same limits and mutation contract |

`SearchOptions.Limit` fixes the ranked-pool depth. `SearchPageOptions.PageSize`
controls the response size; these are deliberately separate. Internal adaptive
retrieval may make multiple probes within the initial pipeline invocation.
Continuation never increases top-K or sends an expanding exclusion list.

In `ranked_then_id`, the tail need not match the query text, exceed a similarity
threshold, or have an embedding. Score and rerank thresholds select the prefix,
not the tail. This mode is for browsing a filtered collection after seeing its
best selected matches—not for pretending that all catalogue entries are ranked
search matches. Use `ranked` when nonmatching catalogue entries are unwanted.

## Example: filtered, grouped browsing

```go
opts := search.DefaultSearchOptions()
opts.Limit = 200 // fixed retrieval depth, before parent grouping
opts.Types = []string{"VideoFrame"}
opts.Filters = map[string][]string{
    "collection_id": {"summer"},
    "license": {"owned", "licensed"},
}
pageOpts := &search.SearchPageOptions{
    Mode: search.SearchPageRankedThenID,
    PageSize: 50,
    GroupBy: "asset_id",
    Scope: "viewer-123/acl-revision-7", // derive from trusted authorization state
}

for {
    page, err := svc.SearchPage(ctx, "sunset beach", queryEmbedding, opts, pageOpts)
    if err != nil {
        return err // handle expiry/invalidation by restarting the whole query
    }
    for _, hit := range page.Results {
        // hit.ID is the representative frame/passage node ID.
        // hit.GroupKey is the logical parent asset ID.
        // hit.Phase == "catalog" means UNRANKED, not a zero-relevance match.
        // Authorize and load full nodes separately when needed.
        _ = hit
    }
    if page.NextCursor == "" {
        break
    }
    pageOpts.Cursor = page.NextCursor
}
```

For an exact metadata-only browse, use `SearchPageID` and pass an empty query
and nil embedding. Ranking options are irrelevant to ID-only membership but are
still bound into the request identity; keep them unchanged.

Repeat the query, embedding, search options, mode, group property, and trusted
scope on every page. The API copies caller-owned options, maps, slices, and
pointer values. Metadata values and type sets are canonically sorted and
deduplicated for cursor identity; the caller's inputs are not mutated. Page
size may change. Reusing a cursor with the same page size returns the same page
and does not advance a server-side iterator.

Do not call again with an empty cursor after the last page: that starts a new
population. Independent first requests may select different ANN shortlists;
stability is guaranteed within an admitted continuation, not across independent
approximate searches.

## Grouping passages or frames into assets

`GroupBy` must name a flat property containing a nonempty string on every
eligible hit node. It is not a graph traversal or parent join. Filters apply to
the frame/passage node, so denormalize required parent metadata onto these nodes
or index/search parent asset nodes directly.

Grouping occurs **before paging**, but **after bounded retrieval**. A budget of
200 frames does not promise 200 distinct parent assets. In `ranked_then_id`, a
parent missed by this ranked budget still appears in the catalogue tail.

A parent with a selected ranked hit uses its highest-scoring selected hit as
representative; equal scores choose the lowest node ID. A parent with no
selected hit uses its lowest eligible node ID. Ranked parents tie-break by
parent ID; tail parents are ordered by parent ID. Returned `id` remains the
representative node ID; `group_key` carries the parent identity.

Missing, empty, non-string, or invalid UTF-8 grouping keys fail the initial
request. They are not silently skipped or placed in one anonymous group. With
grouping enabled, counts refer to distinct parents, not frames.

## Response semantics

`results` contains lightweight descriptors, never full properties or embeddings.
Each descriptor explicitly identifies `phase` as `ranked` or `catalog`. Catalogue
scores are zero placeholders and have no relevance interpretation.

`total` is the size of this fixed declared population. It is **not** an estimate
of all potential ANN matches. `eligible_count` is present only for complete
filtered-collection modes and is the number of logical members admitted by the
initial scan. Filters use the existing property semantics: AND across keys, OR
within values, scalar/array string comparison, and no constraint for empty value
lists. Type matching follows the existing case-normalized label/type-property
semantics. Suppressed, decay-filtered, and non-current temporal nodes are
excluded at initial materialisation.

`ranked_count` is the number of selected ranked members after grouping and
eligibility checks. `ranked_pool_exhausted` becomes true at the prefix boundary;
it can be true while `exhausted` is false and catalogue pages remain.

`exhausted` means this declared population is finished. `collection_exhausted`
can become true only for the two full collection modes. A short or empty ANN
result in `ranked` **never** sets `collection_exhausted` or supplies an
`eligible_count`.

`completion` is one of `more_results`, `candidate_pool_exhausted`, or
`eligible_population_exhausted`. `candidate_limit` reports the configured,
resolved adaptive retrieval ceiling, not a count of vectors actually visited.
`vector_candidate_limit` and `bm25_candidate_limit` report the effective branch
ceilings, including the compressed-vector profile's bound. Each branch's
`*_stop_reason` reports `target_reached`, `short_response`, `candidate_limit`, or
`adaptive_disabled`. A short ANN response therefore has a distinct reason from
stopping at the candidate limit; neither establishes exhaustive ANN recall.
`total_candidates` and `fallback_triggered` retain the initial retrieval's
diagnostics. These values remain fixed across pages.

## Consistency and invalidation

The population freezes IDs, groups, scores, and order—not a transactional MVCC
snapshot of node properties. Full nodes loaded later can have changed and must
be independently authorized. Eligibility is evaluated at initial
materialisation; time-dependent policies do not run again on every page.

Indexing operations invalidate the continuation generation both **before** and
**after** mutation. An initial build spanning a tracked mutation fails instead
of publishing a mixed population. Node indexing/removal, index builds, vector
clearing/property-index removal, reranker changes, decay-filter setter calls,
and changed index master flags are hooked. `Close` permanently ends admission.
No-op `SetReranker` and unchanged master-flag assignments do not invalidate.

Built-in Badger and memory storage publish database-scoped graph revisions;
namespace, WAL, tracing and async wrappers preserve them. Async revisions also
cover visible buffered changes before disk flush. Continuation checks these
revisions when building/returning pages, so completed creates, updates, deletes,
bulk operations, relationship mutations and namespace-prefix deletions invalidate retained populations
without waiting for deferred indexing callbacks. Writes in another database do
not invalidate this database's cursors. This is a change-detection contract, not
an immutable transactional snapshot.

Custom/external engines without `storage.GraphMutationVersionProvider`, and
changes inside captured policy closures, still need explicit coordination.
Bracket the entire storage-plus-index change:

```go
finish := svc.BeginSearchContinuationMutation()
defer finish()

if err := engine.UpdateNode(node); err != nil {
    return err
}
return svc.IndexNode(node) // nested bracketing is safe
```

The explicit contract also applies to external policy changes or low-level
custom mutation paths that do not publish graph revisions. Unreported changes
are not promised snapshot consistency. No database transaction is retained
while a client reads successive pages or calls an external reranker.

Cursors are scoped to one service instance/database and a mandatory caller
scope. Use a trusted authenticated principal plus authorization-policy revision;
never accept a user-supplied scope as proof of permission. The core performs no
independent user authorization. Recheck authorization before every API call and
before hydrating nodes.

Cursors use a random session ID and per-service HMAC-SHA256 signing key. They
contain no query, filters, embeddings, or node IDs. They are signed, not an
encrypted data export. Query identity uses canonical JSON plus SHA-256 rather
than delimiter-concatenated cache keys. Tampered, cross-service, changed-query,
and changed-scope cursors fail explicitly.

Restarting/replacing the service loses the signing key and sessions. A
multi-instance deployment needs affinity to the owning instance; shared cursor
storage is not implemented. Lifetime is fixed from initial admission and is not
extended by reading pages. Final pages remain replayable until expiry, explicit
release, mutation, or close.

Use `errors.Is` with `ErrSearchCursorInvalid`, `ErrSearchCursorMismatch`,
`ErrSearchCursorExpired`, `ErrSearchCursorInvalidated`,
`ErrSearchContinuationLimit`, `ErrSearchContinuationClosed`, or
`ErrSearchPageRequest`. On invalidation/expiry, discard the previous browsing
population and restart with an empty cursor. Do not append a restarted
population to the previous pages and claim duplicate-free continuity.

`ReleaseSearchPage` accepts the same request with any issued cursor from the
session and frees it early. Releasing one cursor ends replay for the whole
session.

## Resource limits and performance

Defaults are a five-minute fixed TTL, 32 retained sessions, 100,000 logical
members per session, 64 MiB total retained descriptor accounting, 32 MiB per
initial-build accounting, one simultaneous build, at most 500 results per page,
5,000 ranked candidates (also limited by the existing engine cap), and 1,000,000
scanned nodes. All scanned nodes count, including filter rejects. Excessive
ranked depth, scan work, session count, member count, or byte accounting fails
with an explicit error; no exact population is silently truncated. One-page
complete populations do not occupy a retained session.

```go
err := svc.ConfigureSearchContinuation(search.SearchContinuationConfig{
    TTL: 20 * time.Minute,
    MaxSessions: 16,
    MaxResults: 100000,
}) // zero fields inherit defaults; reconfiguration invalidates existing cursors
```

The full collection modes pay for one initial scan and sort; they are not
storage keyset cursors. Only ID/score descriptors survive admission. Byte limits
are conservative descriptor/string/map accounting, **not** a bound on total
process RSS or the storage engine's own allocations. In particular,
`StreamNodesWithFallback` materializes all nodes on engines without streaming
support; inherited wrappers may also buffer. Measure the intended engine path
before using this for large catalogues. This is a bounded materialisation
implementation, not an unbounded export solution.

No background worker is created. Expired entries are reclaimed lazily on
operations, mutation, configuration, or close; an idle service can retain
expired descriptors until such an operation. Requests are bounded by explicit
admission controls rather than evicting another caller's live population.

MMR is explicitly rejected: its diversity ordering is not descending
score/ID ordering. Reranking inherits existing `Search` behavior and runs only
inside the initial ranked retrieval where that search path supports it. The
unscored tail is never reranked. Later pages do not rerun ANN, BM25, fusion,
reranking, or storage scans, although repeated request canonicalization still
costs work proportional to the request size.
