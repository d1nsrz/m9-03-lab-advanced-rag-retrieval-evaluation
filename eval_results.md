# Lab 3 — Evaluation Results

Embedding backend: `local`. Eval set: 5 questions, one with the exact term `0x80070005`.

## Comparison table

| Setup | Retrieval hit rate | Faithfulness (LLM judge) |
|---|---|---|
| baseline (dense) | 100% | n/a (set GOOGLE_API_KEY) |
| upgraded (hybrid RRF) | 100% | n/a (set GOOGLE_API_KEY) |

## Per-question retrieval (expected id must be in the set)

| Question | Expected | baseline hit | hybrid hit |
|---|---|---|---|
| How long do I have to get a full refund? | kb-04 | ✅ | ✅ |
| What does error 0x80070005 mean? | kb-08 | ✅ | ✅ |
| How do I cancel my subscription and stop billing? | kb-05 | ✅ | ✅ |
| When are employees allowed to park in lot B? | kb-01 | ✅ | ✅ |
| How do I reset my password? | kb-07 | ✅ | ✅ |

## Conclusion

Hit rate was **flat**: on this tiny, well-separated KB dense retrieval already finds every expected passage, so hybrid had no room to help. On a larger corpus with more lexical collisions the exact-term win shows.

Faithfulness is judged by an LLM and needs `GOOGLE_API_KEY`; with the key set, the table above fills in both columns for a fair side-by-side.
