# Search Continuation

NornicDB exposes one durable START/PULL/DISCARD contract through HTTP, native
gRPC, and the existing `db.retrieve` Cypher procedure. The opaque qid is shared
by these adapters when they use the same server process, authenticated
principal, and canonical database.

## Operations

| Operation | Fields                 | Result                                       |
| --------- | ---------------------- | -------------------------------------------- |
| START     | query/options plus `n` | First page and a qid when more results exist |
| PULL      | `qid`, `n`             | Deterministic page at the token position     |
| DISCARD   | `qid`, `discard: true` | Releases the retained stream                 |

`n` is a page size. In `ranked` mode, `limit` is the initial retrieval depth,
not a lifetime ceiling. `max_results` is the optional explicit lifetime
ceiling. An exhausted response has `has_more: false` and no next qid.

## Modes

| Mode             | Population                                         | Order                          |
| ---------------- | -------------------------------------------------- | ------------------------------ |
| `ranked`         | Progressively discovered canonical ranked hits     | Append-only ranked expansions  |
| `ranked_then_id` | Ranked prefix, then every other eligible result    | Ranked prefix, then logical ID |
| `id`             | Every eligible result without embedding or ranking | Logical ID                     |

`ranked_limit` fixes the ranked prefix in `ranked_then_id`. `group_by` names a
flat property containing a nonempty UTF-8 string. Grouping happens before
pagination: one logical asset consumes one page slot, while its ordered
`passages` collection retains matching child nodes.

`ranked_pool_exhausted` is conservative. It is `true` for `id` mode and when a
ranked branch returns fewer rows than its requested depth. It remains `false`
when `ranked_limit` is filled because more ranked candidates may exist.

## HTTP

```http
POST /nornicdb/search
Content-Type: application/json

{
  "database": "nornic",
  "query": "sunset beach",
  "labels": ["Image"],
  "mode": "ranked_then_id",
  "group_by": "asset_id",
  "limit": 500,
  "ranked_limit": 5000,
  "n": 50
}
```

Pull and discard use the same endpoint:

```json
{ "qid": "opaque-signed-token", "n": 50 }
```

```json
{ "qid": "opaque-signed-token", "discard": true }
```

When no continuation fields are supplied, HTTP preserves its legacy array
response. A continuation request returns an object containing `results`, `qid`,
`has_more`, `position`, `returned`, `discovered`, `total`, `expires_at`,
`search_method`, `fallback_triggered`, `mode`, `ranked_count`,
`eligible_count`, `ranked_pool_exhausted`, `collection_exhausted`, and
`completion`.

## Native gRPC

Set `SearchTextRequest.n` to start. Send `SearchTextResponse.qid` in a later
request with a new `n`, or set `discard`. The additive request fields are
`qid`, `n`, `discard`, `max_results`, `mode`, `group_by`, and `ranked_limit`.
`SearchHit` adds `phase`, `group_key`, and repeated `passages`.

## Cypher and Bolt drivers

Continuation-enabled `db.retrieve` calls return one `page` column:

```cypher
CALL db.retrieve({query: $query, mode: 'ranked', limit: 50, n: 10})
YIELD page RETURN page
```

Resume after reconnecting with any standard Neo4j driver:

```cypher
CALL db.retrieve({qid: $qid, n: 10}) YIELD page RETURN page
```

Ordinary `db.retrieve` calls without continuation options retain their existing
`node`, `score`, `rrf_score`, `vector_rank`, `bm25_rank`, `search_method`, and
`fallback_triggered` columns. Bolt's standard numeric qid remains local to a
connection and transaction; it is not the opaque durable qid. A continuation
START also publishes the opaque token as additive `durable_qid` metadata on
Bolt's `RUN` `SUCCESS` message when more results exist.

## Security, consistency, and lifetime

Qids are signed, fixed-expiry, process-local tokens. Registry state stores a
keyed owner/database digest, not credentials. Pull and discard require the same
validated principal and canonical database. A different server instance cannot
resume a process-local qid, so multi-instance deployments require affinity.

Ranked streams preserve emitted membership, order, and score metadata while
hydrating current node values. Complete `id` and `ranked_then_id` populations
also bind the graph mutation revision and continuation policy generation; a
change invalidates the qid rather than returning an inexact eligible count.

Complete builds fail instead of truncating when scan, member, passage, retained
descriptor byte, build-duration, or concurrent-build limits are exceeded.
The shared registry additionally enforces global and per-owner active-stream
and reported retained-byte limits. Complete streams report their compact
descriptor bytes to this admission layer; streams that do not implement byte
reporting are still governed by stream-count limits.

Complete builds prefer the storage streaming interface. Engines without it use
`AllNodes` fallback, whose temporary full node slice is outside retained
descriptor admission. On the 20,000-node benchmark fixture (Apple M3 Max), the
native path used about 29.9 MB/op versus 37.6 MB/op for fallback; operators
using a fallback engine must budget for that additional build-time memory.
