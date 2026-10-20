# Claim-Level Judge for i-MedRAG

A verification layer for iterative medical retrieval-augmented generation (RAG).
It sits on top of [i-MedRAG](https://arxiv.org/abs/2408.00727) (Xiong et al., 2025) and checks every intermediate
answer **claim by claim** against the retrieved documents before the answer is
written into the iterative history. Only claims that are *contradicted* by the
documents are discarded. An optional adaptive planner also decides how many
follow-up queries each round needs.

This repository accompanies a Persian-language master's thesis (IUST-Thesis LaTeX class, `xepersian`).

## Motivation

In the original i-MedRAG loop, the answer to each follow-up query is appended to
the history without any check. A single wrong intermediate answer can therefore
mislead all later rounds. The judge is inserted exactly at this point, between
`medrag_answer` and the history update, without changing the retriever or the
answering model.

## Method overview

For every follow-up query `q`:

1. **Answer and extract claims.** `medrag_answer` produces a raw answer `a` and
   the retrieved documents `D_q`. A model call decomposes `a` into atomic claims
   (JSON with a `claims` key; sentence splitting is the fallback).
2. **Drop meta claims.** Claims about the documents themselves (for example
   "Document 3 exists") are removed by regex before judging. They are useless,
   not false.
3. **Judge each claim** independently against `D_q` with one of three verdicts:
   `supported`, `contradicted`, or `unverifiable`. Unparseable judge output
   defaults to `unverifiable`, never `contradicted`.
4. **History update policy.**
   - no contradicted claim: the raw answer `a` is kept unchanged;
   - some contradicted and some remaining claims: only the remaining claims are joined;
   - all claims contradicted: the query-answer pair is not added to the history.
5. **Adaptive query planner (optional).** Before each round a small model call
   chooses `n_i` in `{1..n_max}` (round 1) or `{0..n_max}` (later rounds).
   `n_i = 0` ends the loop and triggers the final-answer round.
6. **Final answer.** Same as the base i-MedRAG, but built from the refined history.
7. **Audit log.** All claims, labels and reasons are logged (for example
   `claims.json`), together with `jmr_plan_log` and `jmr_queries_per_round`.

## Configurations

| Config | Queries per round | Claim judge | Entry point |
|--------|-------------------|-------------|-------------|
| `S0` baseline i-MedRAG | fixed (`n`) | no | `i_medrag_answer` |
| `S1` adaptive, no judge | adaptive (`<= n_max`) | no | `judged_i_medrag_answer` with judging disabled |
| `S2` adaptive + judge (full method) | adaptive (`<= n_max`) | yes | `judged_i_medrag_answer` |

`adaptive_queries=False` restores the original fixed-`n` behaviour so the
planner can be evaluated separately.

## Experimental setup

| Item | Value |
|------|-------|
| Dataset | MedQA, USMLE subset (4 options), first 500 questions |
| LLM | Llama-3.1-8B-Instruct via Chabokan (`LLM_NAME = "Chabokan/meta-llama/llama-3.1-8b-instruct"`) |
| Retriever / corpus | MedCPT, Textbooks |
| Retrieved chunks | `k = 10` |
| Rounds | `m = 2` |
| Queries | `S0`: 4 per round (8 total); `S1`/`S2`: `n_max = 4` per round |
| Temperature | 0 |

## Results (500 questions)

| Config | Correct | Accuracy | Mean queries / question |
|--------|---------|----------|-------------------------|
| `S0` | 368 | 0.736 | 8.00 |
| `S1` | 376 | 0.752 | 3.65 |
| `S2` | 395 | 0.790 | 3.57 |

- `S2` vs `S0`: accuracy difference +0.054, exact McNemar test `p = 0.0197`.
- `S0` vs `S1` (`p = 0.4884`) and `S1` vs `S2` (`p = 0.0871`) are not significant at 0.05.
- The judge labelled 25,998 claims in `S2`: 93.39% supported, 5.79% unverifiable, 0.82% contradicted (and removed).
- The adaptive planner cuts the number of follow-up queries by about 55% relative to the fixed baseline.


## Reference

Xiong, G., et al. *Improving Retrieval-Augmented Generation in Medicine with
Iterative Follow-up Questions* (i-MedRAG), 2025.
