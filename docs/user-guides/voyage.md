# Native Voyage retrieval

NornicDB supports Voyage reranking, contextualized document embeddings and multimodal embeddings through its existing search service and background embedding worker. Existing embedding and reranking providers retain their interfaces.

## Configure a server

Set credentials through your deployment's secret mechanism. These example environment settings select a text database with provider-managed document chunking:

```text
NORNICDB_EMBEDDING_ENABLED=true
NORNICDB_EMBEDDING_PROVIDER=voyage-context
NORNICDB_EMBEDDING_MODEL=voyage-context-4
NORNICDB_EMBEDDING_DIMENSIONS=2048
NORNICDB_EMBEDDING_API_URL=https://api.voyageai.com
NORNICDB_EMBEDDING_API_KEY=<secret>
NORNICDB_EMBED_CHUNK_SIZE=512
NORNICDB_EMBED_CHUNK_OVERLAP=0
NORNICDB_EMBED_AUTO_CHUNKING=true
NORNICDB_SEARCH_RERANK_ENABLED=true
NORNICDB_SEARCH_RERANK_PROVIDER=voyage
NORNICDB_SEARCH_RERANK_MODEL=rerank-2.5
NORNICDB_SEARCH_RERANK_API_URL=https://api.voyageai.com/v1/rerank
NORNICDB_SEARCH_RERANK_API_KEY=<secret>
NORNICDB_SEARCH_RERANK_FAILURE_POLICY=error
NORNICDB_SEARCH_RERANK_TRUNCATION=false
```

Models and dimensions are configurable. The native Go provider defaults to `voyage-context-4` or `voyage-multimodal-3.5` and 2,048 dimensions; specify these values explicitly in server configuration to override its existing defaults.

Contextualized chunk size and overlap are token counts. The server defaults are 8,192 tokens per chunk, 50 tokens of overlap and automatic chunking enabled; zero overlap is supported. Setting automatic chunking to `false` sends the complete document as one contextualized input, omitting chunk size and overlap. That document must fit the provider's single-input limit. Vector output uses float32, matching NornicDB's vector storage and indexes.

| Control | YAML setting | Per-database setting |
| --- | --- | --- |
| Chunk token size | `embedding_worker.chunk_size` | `db.nornic.embed.chunk.size` |
| Overlap tokens | `embedding_worker.chunk_overlap` | `db.nornic.embed.chunk.overlap` |
| Context automatic chunking | `embedding_worker.auto_chunking` | `db.nornic.embed.auto.chunking` |
| Vector dimensions | `embedding.dimensions` | `db.nornic.embedding.dimensions` |

The environment settings above configure the same controls. Chunk controls apply to the contextualized provider; multimodal generation uses its own structured input. Supported dimensions depend on the selected Voyage model.

One process can use separate logical databases for text and visual retrieval. For example, a YAML configuration can define these overrides for a database created with `CREATE DATABASE visual`:

```yaml
databases:
  visual:
    db.nornic.embedding.provider: voyage-multimodal
    db.nornic.embedding.model: voyage-multimodal-3.5
    db.nornic.embedding.dimensions: "2048"
    db.nornic.embedding.api.url: https://api.voyageai.com
    db.nornic.search.rerank.enabled: "false"
```

The existing database configuration API supports the same settings. YAML database overrides supply initial defaults; later stored administrator changes remain authoritative. Query generation and the background worker resolve the selected database's provider. A model-space identity includes endpoint, API family, model, dimensions and vector representation. Equal dimensions do not make text and visual vectors compatible.

Changing a model does not silently regenerate existing accepted results. Search excludes a stored native vector whose space differs from the configured space. Use the existing explicit embedding-clear/regeneration operation when changing the population to a new model. Stored job status exposes the result's space; `current_result` describes its source currentness.

## Store documents and images

Create or update ordinary nodes through Bolt or the Neo4j HTTP API. The worker reads the configured embedding properties and sends the complete document to Voyage with the configured automatic chunking option. It persists all returned vectors and their ordered, complete chunk texts on the source node.

```cypher
CREATE (d:Document {content: $wholeDocument, source_id: $sourceId})
RETURN elementId(d) AS id
```

For a visual database, `_embedding_content` contains a JSON array of structured parts. A JSON string works with clients whose property model cannot store arrays of maps:

```cypher
CREATE (i:Image {_embedding_content: $partsJson, source_id: $sourceId})
RETURN elementId(i) AS id
```

Example parameter values:

```json
[{"type":"image_url","image_url":"https://example.com/image.png"}]
```

```json
[{"type":"image_base64","image_base64":"data:image/png;base64,..."}]
```

Text and image parts can be interleaved. Ordinary URL properties are not implicitly treated as images. The provider receives structured image input; NornicDB does not fetch or transcribe media. Visual text queries use the multimodal model with query purpose. Contextualized queries also use explicit query purpose and are sent whole.

`WITH EMBEDDING` uses the same complete-document result representation for explicit synchronous generation. Its failure fails the enclosing write; background jobs instead expose persisted attempt state.

## Retrieve supporting passages and rerank

```cypher
CALL db.rretrieve({query: $query, limit: 10})
```

For native contextualized results, the additional `supporting_passages` column contains the exact provider-returned text, parent `node_id`, zero-based `chunk_index`, `matched_by`, `space` and `source_fingerprint`. Dense retrieval retains the winning chunk; lexical retrieval selects supporting text from the provider's chunks with the configured BM25 implementation. A hybrid result can include both supporting chunks. The source properties remain unchanged.

Native reranking runs once after candidate retrieval or query-result fusion. It receives complete supporting passages when available, or the existing searchable text for ordinary candidates. It preserves candidate IDs and original content. Provider ordering and scores determine the result; no flat-score heuristic replaces them. The default candidate pool is 100, with a maximum of 1,000 submitted candidates. Provider input limits produce an error unless the caller explicitly permits truncation.

Explicit candidate reranking:

```cypher
CALL db.rerank({
  query: $query,
  candidates: $candidates,
  limit: 10,
  rerankFailurePolicy: 'error',
  rerankTruncation: false
})
```

`rerankFailurePolicy: 'original'` explicitly returns the original candidate ordering on provider failure and reports `status: 'fallback'`. The default `error` policy returns failure. `rerankTopK`, `rerankMinScore`, `rerankTruncation` and `rerankFailurePolicy` are accepted by the retrieval request. Native result rows include a `rerank` report with submitted/returned counts and sanitized provider diagnostics. An empty successful reranking result remains empty.

When the default policy returns an error instead of result rows, the error text preserves the known provider diagnostics as `metadata={...}`: HTTP status, attempt count, and available request ID, model, usage and retry timing. These are the same sanitized fields available on Go `voyage.Error.Metadata`; credentials, submitted content, raw response bodies and transport URLs are excluded. Missing provider fields remain absent rather than being invented.

`POST /nornicdb/search` accepts `database`, `query`, `limit`, `rerank_top_k`, `rerank_min_score`, `rerank_truncation` and `rerank_failure_policy`. Its result objects carry passages and rerank reports; `X-NornicDB-Rerank` also carries the report when there are no result rows. Bolt completion metadata and Neo4j HTTP result metadata retain native rerank reports. Native gRPC search exposes typed supporting passages and a `nornicdb-rerank` response trailer. MCP discovery retains supporting passages; MCP Cypher calls use the same retrieval procedures.

## Observe and control background work

```cypher
CALL db.embedding.status({nodeId: $id}) YIELD status
CALL db.embedding.control({nodeId: $id, action: 'pause'}) YIELD status
CALL db.embedding.control({nodeId: $id, action: 'resume'}) YIELD status
```

Control actions are `pause`, `cancel`, `resume` and `retry`. `retry` applies to terminal failed work. Pause and Cancel invalidate an in-flight operation's publication identity; an already accepted current result is retained. Resume permits fresh work when required. Status/control calls use existing database access and read/write permissions.

Status reports pending, running, completed, paused, cancelled or failed work, plus control, attempt count/history, operation ID, retry time, error code and available provider request/status/usage metadata. Rate limits and retryable provider failures use the existing worker retry budget, with delayed attempts retained in its durable pending index. Terminal failures remain visible. Source edits start a new source-bound attempt budget. Restart retains completed results and pending attempts.

Publication compares the current source and operation inside the storage update. A result generated for an edited, deleted, paused or cancelled source cannot overwrite the current state. Local storage wrappers, WAL recovery and the replicated write path transport the same complete state; followers do not generate embeddings to reconstruct a replicated result.

## Application responsibilities

An application can delegate provider document chunking, query embedding, vector persistence/indexing, supporting-passage retrieval and native reranking to NornicDB. It still owns source ingestion and extraction, source IDs/revisions and eligibility, factual filters, its user permissions, and mapping its processing controls to the native job controls. This capability does not extract audio/video, invent image captions, or replace application-level pagination and population accounting.

Go clients use `embed.ManagedProvider`/`PrepareManagedNode`/`EmbedDocument` for documents and `embed.QueryVector` for queries. Low-level search users configure `Service.SetEmbeddingSpace(provider.EmbeddingSpace().Key())` before building native indexes. Legacy `Embed` and `EmbedBatch` on a native Voyage provider return an explicit-purpose error, preventing accidental query/document mixing or loss of all but the first chunk.
