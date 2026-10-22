# PageGlance retrieval evaluation — rag-eval-v1

- Embedding profile: `sentence_transformers:sentence-transformers/all-MiniLM-L6-v2:dimensions=384`
- Queries: 7; retrieval cutoff: top-5; repeats: 3
- Scope isolation is checked against each case's declared document/source filter.
- Unsupported cases are excluded from relevance averages; non-empty retrieval is reported separately.
- This report evaluates retrieval only. It does not call Groq or score generated answers.

| Method | Supported queries | Recall@5 | MRR@5 | nDCG@5 | Median ms | p95 ms | Unsupported with candidates | Scope violations | Citation recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vector | 6 | 1.000 | 1.000 | 1.000 | 38.27 | 39.18 | 1 | 0 | 1.000 |
| lexical | 6 | 0.667 | 0.667 | 0.667 | 5.58 | 6.82 | 1 | 0 | 0.667 |
| rrf | 6 | 1.000 | 0.875 | 0.905 | 42.75 | 45.78 | 1 | 0 | 1.000 |

## PostgreSQL query plan

Default planner:

```text
Limit
  ->  Seq Scan on document_chunks
        Filter: (to_tsvector('simple'::regconfig, text) @@ '''pageglance'' & ''local'' & ''embeddings'''::tsquery)
```

GIN-index eligibility probe with sequential scans disabled:

```text
Limit
  ->  Bitmap Heap Scan on document_chunks
        Recheck Cond: (to_tsvector('simple'::regconfig, text) @@ '''pageglance'' & ''local'' & ''embeddings'''::tsquery)
        ->  Bitmap Index Scan on ix_document_chunks_text_fts
              Index Cond: (to_tsvector('simple'::regconfig, text) @@ '''pageglance'' & ''local'' & ''embeddings'''::tsquery)
```
