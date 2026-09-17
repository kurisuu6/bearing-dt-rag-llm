# Bearing DT + SKF RAG + LLM Agent

This repository implements a bearing diagnosis question-answering system that combines:

- a lightweight digital twin (DT) built from the IMS bearing dataset,
- a SKF handbook retrieval-augmented generation (RAG) knowledge base,
- a router for `unrelated`, `dt_only`, `rag_only`, and `dt_rag` questions,
- LLM-based DT query planning and final answer generation,
- evaluation pipelines for DT, router, RAG-only, and DT+RAG experiments.

The current main experimental setting uses the IMS bearing dataset for DT queries and SKF handbook chapters 1-11 for RAG.

## Data and Source Material

The repository does not include the IMS raw dataset, `SKF.pdf`, parsed SKF
chapters, generated RAG chunks, vector indexes, DT databases, or manuscript
files. These materials must be obtained or generated locally according to their respective terms of use. The repository contains only code, reviewed question data, and metric-only experiment summaries.

## Repository Layout

```text
DT/
  build_ims_dt_database.py          Build IMS feature database
  build_dt_benchmark.py             Build DT benchmark questions/references
  dt_query_tools.py                 Deterministic DT query tools
  dt_llm_planner.py                 LLM planner for DT query parameters
  evaluate_dt_benchmark.py          Rule-based DT benchmark evaluation
  evaluate_dt_benchmark_llm.py      LLM-planner DT benchmark evaluation
  eval/
  README.md

SKF-RAG/
  parse_skf_with_llamaparse.py      Parse SKF PDF into markdown JSON
  parse_skf_chapters_02_12.py       Batch parser for SKF chapters
  build_skf_rag_chunks_llamaindex.py
                                    Build RAG chunks from parsed SKF JSON
  build_skf_1_11_rag.py             Build combined chapter 1-11 RAG corpus
  build_llamaindex_vector_index.py  Build LlamaIndex vector index

agent/
  bearing_agent.py                  Integrated router + DT + RAG agent
  router.py                         Rule router
  llm_router.py                     LLM and hybrid router
  http_embedding.py                 HTTP embedding adapter
  evaluate_router.py                Router evaluation
  evaluate_rag_only_with_ragas.py   RAG-only evaluation
  evaluate_dt_rag_with_ragas.py     DT+RAG evaluation
  evaluate_agent.py                 Integrated agent evaluation
  trace_question_flow.py            Single-question trace/debug tool
  check_eval_env.py                 Environment-variable check
  build_rag_only_gold_chunk_dataset.py
  build_dt_rag_gold_chunk_dataset.py
  eval/                             Active datasets and experiment results

docs/                               Project notes
requirements.txt
```

## Environment

Python 3.10 is recommended.

```bash
conda create -n SKF-rag python=3.10 -y
conda activate SKF-rag
pip install -r requirements.txt
```

## Platform Notes

The project was developed and tested in a macOS + Conda + Python 3.10 environment. The same Python scripts and shell commands should also work on Linux.

Windows is not the primary tested environment. Most Python code should still be portable, but Windows users may need to adapt shell syntax, for example replacing `export VAR=value` and line continuation `\` with PowerShell equivalents.

Recommended environment for reproducing the experiments:

```text
macOS or Linux
Python 3.10
Conda environment
External HTTP services for embedding, reranking, and llama3.3:70b if using the same setup
```

Create a local `.env` from `.env.example` or export variables in your shell. Do not commit real keys or service URLs.

```bash
# OpenAI-compatible API for GPT models or relay services
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"

# LlamaParse, only needed if parsing SKF.pdf again
export LLAMA_CLOUD_API_KEY="your_llamacloud_key"

# HTTP embedding service, currently used for bge-m3
export EMBEDDING_URL="http://your-embedding-service"

# HTTP reranker service, currently used for bge-reranker-v2-m3
export RERANKER_URL="http://your-reranker-service"

# OpenAI-compatible Ollama endpoint, used for llama3.3:70b
export OLLAMA_BASE_URL="http://your-ollama-service"
export OLLAMA_API_KEY="your_ollama_gateway_key"
```

For OpenAI-compatible Ollama calls, use:

```text
${OLLAMA_BASE_URL}/v1
```

## Digital Twin Pipeline

Build the IMS DT database:

```bash
python3 DT/build_ims_dt_database.py \
  --data-root DT/IMS \
  --output-dir DT/outputs \
  --db DT/outputs/ims_dt.db
```

Main outputs:

```text
DT/outputs/ims_dt.db
DT/outputs/ims_feature_records.csv
DT/outputs/ims_bearing_snapshots.csv
DT/outputs/ims_latest_bearing_states.csv
DT/outputs/ims_dt_summary.json
```

The DT database stores extracted bearing features, health states, latest-state snapshots, and trend-related records. In question answering, "current" or "latest" means the latest available IMS record in the offline DT database, not the real-world date.

Example DT query:

```bash
python3 DT/dt_query_tools.py latest-state \
  --experiment 1st_test \
  --bearing bearing_3
```

## SKF RAG Pipeline

Before building the vector index, the SKF PDF is parsed into structured JSON files using LlamaParse. The parser extracts page-level Markdown content and tables while preserving the original page numbers.

```bash
python3 SKF-RAG/parse_skf_with_llamaparse.py \
  --pdf SKF-RAG/SKF.pdf \
  --page-range "PAGE_START-PAGE_END" \
  --output SKF-RAG/chapter_XX/chapter_XX_llama_raw.json
```  

The parsed chapter files are then used to construct RAG chunks. The chunking process preserves the heading structure and page metadata, while tables are stored as independent chunks. Text chunks use a default size of 650 tokens with a 90-token overlap.

```bash
python3 SKF-RAG/build_skf_rag_chunks_llamaindex.py \
  --input SKF-RAG/chapter_XX/chapter_XX_llama_cleaned.json \
  --output SKF-RAG/chapter_XX/chapter_XX_rag_chunks_llamaindex.json \
  --chapter "CHAPTER_NAME" \
  --chunk-size 650 \
  --chunk-overlap 90
```

The preferred vector index uses the HTTP embedding service:

```text
SKF-RAG/chapters_01_11/llamaindex_vector_index_bge_m3/
SKF-RAG/chapters_01_11/llamaindex_vector_index_bge_m3_manifest.json
```

Rebuild the vector index:

```bash
python3 SKF-RAG/build_llamaindex_vector_index.py \
  --input SKF-RAG/chapters_01_11/skf_chapters_01_11_rag_chunks_llamaindex.json \
  --persist-dir SKF-RAG/chapters_01_11/llamaindex_vector_index_bge_m3 \
  --manifest SKF-RAG/chapters_01_11/llamaindex_vector_index_bge_m3_manifest.json \
  --embedding-provider http \
  --embedding-api-url "$EMBEDDING_URL"
```

The agent supports four retrieval modes:

```text
bm25              lexical sparse retrieval over SKF chunks
dense             vector retrieval over the LlamaIndex index
hybrid            fusion of BM25 and dense retrieval
hybrid_reranker   hybrid candidates reranked by the HTTP reranker service
```

For `hybrid_reranker`, `--retrieval-top-k` controls the candidate pool before reranking, while `--top-k` controls how many final chunks are passed to the answer model and evaluator.

## Agent Flow

The integrated agent follows this workflow:

```text
user question
  -> router
  -> selected route
      unrelated: no DT/RAG tool call
      dt_only: query IMS DT database
      rag_only: retrieve SKF chunks and answer
      dt_rag: query DT first, build a DT-augmented RAG query, retrieve SKF chunks, then answer
```

For DT+RAG questions, the system first obtains DT evidence such as health state, abnormal features, trends, and latest feature values. It then builds a DT-augmented retrieval query using:

- the original user question,
- DT evidence,
- the DT+RAG intent, such as fault explanation, fault type identification, or maintenance decision,
- engineering terms related to vibration, bearing damage, lubrication, inspection, and condition monitoring.

The final response is generated from both DT evidence and retrieved SKF evidence.

## Normal Question Answering

Run a full question-answering path with router, DT/RAG tools, and final answer generation:

```bash
python3 agent/bearing_agent.py \
  --question "Why do high RMS and high kurtosis in bearing_3 of 1st_test suggest bearing damage?" \
  --db DT/outputs/ims_dt.db \
  --index-dir SKF-RAG/chapters_01_11/llamaindex_vector_index_bge_m3 \
  --manifest SKF-RAG/chapters_01_11/llamaindex_vector_index_bge_m3_manifest.json \
  --router-mode hybrid \
  --router-model llama3.3:70b \
  --retrieval-mode hybrid_reranker \
  --top-k 5 \
  --retrieval-top-k 20 \
  --reranker-top-k 5 \
  --dt-planner-mode llm \
  --dt-planner-model llama3.3:70b \
  --dt-planner-api-base "$OLLAMA_BASE_URL/v1" \
  --dt-planner-api-key "$OLLAMA_API_KEY" \
  --answer \
  --llm-model llama3.3:70b \
  --llm-api-base "$OLLAMA_BASE_URL/v1" \
  --llm-api-key "$OLLAMA_API_KEY"
```

Trace a single evaluation question through router, DT, RAG, and answer generation:

```bash
python3 agent/trace_question_flow.py \
  --questions agent/eval/agent_english_questions_v2.json \
  --question-id dt_rag_explain_001 \
  --router-mode hybrid \
  --router-model llama3.3:70b \
  --retrieval-mode hybrid_reranker \
  --top-k 5 \
  --retrieval-top-k 20 \
  --reranker-top-k 5 \
  --answer \
  --llm-model llama3.3:70b \
  --llm-api-base "$OLLAMA_BASE_URL/v1" \
  --llm-api-key "$OLLAMA_API_KEY" \
  --output-json agent/eval/trace_dt_rag_explain_001.json \
  --output-md agent/eval/trace_dt_rag_explain_001.md
```

## Evaluation Datasets

Main active evaluation files:

```text
agent/eval/agent_english_questions_v2.json
agent/eval/rag_only_gold_chunk_dataset_v2_reviewed.json
agent/eval/rag_only_reference_answers_gold_v2_reviewed.json
agent/eval/dt_rag_gold_chunk_dataset_v1.json
agent/eval/dt_rag_reference_answers_gold_v1.json
DT/eval/dt_benchmark_v1.json
```

The gold datasets and full evaluation outputs listed above are local
experiment artifacts and are not included in the public repository unless
they are explicitly added after a separate review. The public repository
keeps only the question set and metric-only `*_means.csv` summaries.

The full agent question set contains 140 English questions:

| Route | Count | Meaning |
|---|---:|---|
| `unrelated` | 30 | Outside bearing DT/RAG scope |
| `rag_only` | 50 | SKF manual knowledge only |
| `dt_only` | 30 | IMS DT feature/state/trend query |
| `dt_rag` | 30 | DT evidence plus SKF knowledge |

The RAG-only and DT+RAG gold datasets use one gold SKF chunk per question. Retrieval is evaluated with `Hit@1`, `Hit@5`, and `MRR`; answer quality is evaluated with RAGAS metrics such as `faithfulness`, `answer_relevancy`, and `context_precision`.

## Router Evaluation

Run the three router settings on the full 140-question benchmark:

```bash
python3 agent/evaluate_router.py \
  --questions agent/eval/agent_english_questions_v2.json \
  --router-mode rule \
  --results agent/eval/router_eval_results_full140_rule.json \
  --summary agent/eval/router_eval_summary_full140_rule.csv
```

```bash
python3 agent/evaluate_router.py \
  --questions agent/eval/agent_english_questions_v2.json \
  --router-mode llm \
  --router-model llama3.3:70b \
  --router-timeout 120 \
  --results agent/eval/router_eval_results_full140_llama.json \
  --summary agent/eval/router_eval_summary_full140_llama.csv
```

```bash
python3 agent/evaluate_router.py \
  --questions agent/eval/agent_english_questions_v2.json \
  --router-mode hybrid \
  --router-model llama3.3:70b \
  --router-timeout 120 \
  --results agent/eval/router_eval_results_full140_hybrid_corrected.json \
  --summary agent/eval/router_eval_summary_full140_hybrid_corrected.csv
```

Current router results:

| Router | Total | Accuracy | Error count | Unrelated | RAG-only | DT-only | DT+RAG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule | 140 | 0.9286 | 0 | 30/30 | 44/50 | 28/30 | 28/30 |
| LLM llama3.3:70b | 140 | 0.9714 | 0 | 30/30 | 50/50 | 26/30 | 30/30 |
| Hybrid | 140 | 0.9786 | 0 | 30/30 | 50/50 | 28/30 | 29/30 |

## DT Benchmark

Run the LLM-planner DT benchmark:

```bash
python3 DT/evaluate_dt_benchmark_llm.py \
  --benchmark DT/eval/dt_benchmark_v1.json \
  --db DT/outputs/ims_dt.db \
  --planner-model llama3.3:70b \
  --planner-api-base "$OLLAMA_BASE_URL/v1" \
  --planner-api-key "$OLLAMA_API_KEY" \
  --judge-model llama3.3:70b \
  --judge-api-base "$OLLAMA_BASE_URL/v1" \
  --judge-api-key "$OLLAMA_API_KEY" \
  --results DT/eval/dt_benchmark_results_llama_planner_judge_full_v4.json \
  --summary DT/eval/dt_benchmark_summary_llama_planner_judge_full_v4.csv
```

Current DT benchmark results:

| Scope | Total | Query accuracy | Execution accuracy | Answer accuracy | Joint accuracy |
|---|---:|---:|---:|---:|---:|
| Overall | 30 | 0.9667 | 1.0000 | 0.9333 | 0.9000 |
| Feature | 10 | 1.0000 | 1.0000 | 0.9000 | 0.9000 |
| State | 10 | 0.9000 | 1.0000 | 1.0000 | 0.9000 |
| Trend | 10 | 1.0000 | 1.0000 | 0.9000 | 0.9000 |

## RAG-only Evaluation

RAG-only evaluation uses 50 SKF manual questions and the reviewed single-gold reference file:

```text
agent/eval/rag_only_reference_answers_gold_v2_reviewed.json
```

Example Hybrid + Reranker run:

```bash
python3 agent/evaluate_rag_only_with_ragas.py \
  --questions agent/eval/agent_english_questions_v2.json \
  --references agent/eval/rag_only_reference_answers_gold_v2_reviewed.json \
  --dataset-output agent/eval/rag_only_gold_v2_full_hybrid_reranker_dataset.jsonl \
  --results agent/eval/rag_only_gold_v2_full_hybrid_reranker_results.json \
  --summary agent/eval/rag_only_gold_v2_full_hybrid_reranker_summary.csv \
  --retrieval-mode hybrid_reranker \
  --top-k 5 \
  --retrieval-top-k 20 \
  --reranker-top-k 5 \
  --force-rag-only \
  --answer \
  --llm-model llama3.3:70b \
  --judge-model gpt-4o-mini \
  --ragas-embedding-provider http \
  --ragas-workers 1 \
  --skip-context-recall \
  --skip-hit-at-10
```

Current RAG-only results:

| Retrieval | Hit@1 | Hit@5 | MRR | Faithfulness | Answer relevancy | Context precision |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.5400 | 0.7400 | 0.6267 | 0.8917 | 0.9020 | 0.8877 |
| Dense | 0.5200 | 0.8400 | 0.6390 | 0.9501 | 0.9483 | 0.9149 |
| Hybrid | 0.6000 | 0.8800 | 0.7057 | 0.9148 | 0.9314 | 0.9239 |
| Hybrid + Reranker | 0.6000 | 0.9200 | 0.7180 | 0.9265 | 0.9390 | 0.9660 |

## DT+RAG Evaluation

DT+RAG evaluation uses 30 questions and one gold SKF chunk per question:

```text
agent/eval/dt_rag_reference_answers_gold_v1.json
```

Example Hybrid run:

```bash
python3 agent/evaluate_dt_rag_with_ragas.py \
  --questions agent/eval/agent_english_questions_v2.json \
  --references agent/eval/dt_rag_reference_answers_gold_v1.json \
  --dataset-output agent/eval/dt_rag_gold_v1_dataset_hybrid_top5_retrieval20_rule_after_fix_gptjudge.jsonl \
  --results agent/eval/dt_rag_gold_v1_results_hybrid_top5_retrieval20_rule_after_fix_gptjudge.json \
  --summary agent/eval/dt_rag_gold_v1_summary_hybrid_top5_retrieval20_rule_after_fix_gptjudge.csv \
  --db DT/outputs/ims_dt.db \
  --index-dir SKF-RAG/chapters_01_11/llamaindex_vector_index_bge_m3 \
  --manifest SKF-RAG/chapters_01_11/llamaindex_vector_index_bge_m3_manifest.json \
  --retrieval-mode hybrid \
  --top-k 5 \
  --retrieval-top-k 20 \
  --force-route dt_rag \
  --answer \
  --llm-model llama3.3:70b \
  --llm-api-base "$OLLAMA_BASE_URL/v1" \
  --llm-api-key "$OLLAMA_API_KEY" \
  --judge-model gpt-4o-mini \
  --dt-planner-mode llm \
  --dt-planner-model llama3.3:70b \
  --dt-planner-api-base "$OLLAMA_BASE_URL/v1" \
  --dt-planner-api-key "$OLLAMA_API_KEY" \
  --ragas-embedding-provider http \
  --request-timeout 240 \
  --metric-retry-attempts 2 \
  --ragas-workers 1 \
  --skip-context-recall \
  --skip-hit-at-10
```

Current DT+RAG results:

| Retrieval | Hit@1 | Hit@5 | MRR | Faithfulness | Answer relevancy | Context precision |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.5667 | 0.6000 | 0.5733 | 0.7712 | 0.7014 | 0.9523 |
| Dense | 0.3667 | 0.8000 | 0.5428 | 0.8247 | 0.7166 | 0.9807 |
| Hybrid | 0.6000 | 0.8000 | 0.6478 | 0.8514 | 0.6917 | 0.9827 |
| Hybrid + Intent Reranker | 0.4667 | 0.7667 | 0.5911 | 0.8073 | 0.6838 | 0.9706 |

## Notes

- The current main RAG scope is SKF chapters 1-11.
- Chapter 12 has parsed/chunked artifacts, but it is not part of the current main experimental index.
- RAGAS judge calls can time out when using a large local model through a gateway. Use smaller pilot runs, `--request-timeout`, `--metric-retry-attempts`, and `--ragas-workers 1` when debugging.
- The pydantic warning that appears during some runs is a dependency warning and does not normally affect results.
