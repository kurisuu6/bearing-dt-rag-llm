# Project Structure

The repository is organized by subsystem. The current layout keeps the research workflow explicit: DT construction, SKF RAG construction, agent execution, and evaluation are separated.

```text
bearing-dt-rag-llm/
├── DT/
│   ├── build_ims_dt_database.py        # IMS feature extraction and SQLite DT construction
│   ├── build_dt_benchmark.py           # DT benchmark construction
│   ├── dt_query_tools.py               # Deterministic DT query tools and CLI
│   ├── dt_llm_planner.py               # LLM planner for DT query parameters
│   ├── evaluate_dt_benchmark.py        # Rule-based DT benchmark evaluation
│   ├── evaluate_dt_benchmark_llm.py    # LLM-planner DT benchmark evaluation
│   ├── eval/                           # DT benchmark datasets and results
│   ├── outputs/                        # Generated DT CSV/JSON/SQLite outputs
│   └── IMS/                            # Local raw IMS data, ignored by git
├── SKF-RAG/
│   ├── parse_skf_with_llamaparse.py    # SKF PDF parsing through LlamaParse
│   ├── parse_skf_chapters_02_12.py     # Batch chapter parsing helper
│   ├── build_skf_rag_chunks_llamaindex.py
│   │                                   # Build RAG chunks from parsed SKF JSON
│   ├── build_skf_1_11_rag.py           # Build combined chapter 1-11 RAG corpus
│   ├── build_llamaindex_vector_index.py
│   │                                   # Build LlamaIndex vector indexes
│   ├── chapter_*/                      # Per-chapter parsed JSON and chunks
│   ├── chapters_01_11/                 # Main combined RAG corpus and vector index
│   ├── eval/                           # Older SKF-only evaluation artifacts
│   └── archive/                        # Archived one-off scripts
├── agent/
│   ├── bearing_agent.py                # Integrated router + DT + RAG agent
│   ├── router.py                       # Rule router
│   ├── llm_router.py                   # LLM and hybrid router
│   ├── http_embedding.py               # HTTP embedding adapter
│   ├── evaluate_router.py              # Router evaluation
│   ├── evaluate_rag_only_with_ragas.py # RAG-only evaluation
│   ├── evaluate_dt_rag_with_ragas.py   # DT+RAG evaluation
│   ├── evaluate_agent.py               # Integrated agent evaluation
│   ├── trace_question_flow.py          # Single-question trace/debug tool
│   ├── check_eval_env.py               # Environment-variable check
│   ├── eval/                           # Active benchmark files and experiment results
│   └── archive/                        # Archived one-off scripts
├── configs/                            # Reserved runtime config directory
├── docs/                               # Project documentation
├── tools/                              # Maintenance utilities
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Active Artifacts

Main files currently used by experiments:

```text
agent/eval/agent_english_questions_v2.json
agent/eval/rag_only_reference_answers_gold_v2_reviewed.json
agent/eval/dt_rag_reference_answers_gold_v1.json
DT/eval/dt_benchmark_v1.json
SKF-RAG/chapters_01_11/skf_chapters_01_11_rag_chunks_llamaindex.json
SKF-RAG/chapters_01_11/llamaindex_vector_index_bge_m3/
```

## Archive Policy

Archived files are kept for traceability, not for normal execution. They include smoke tests, older evaluation attempts, temporary review scripts, and superseded experiment outputs.

```text
agent/archive/
agent/eval/archive/
SKF-RAG/archive/
```

For normal project use, start from `README.md`. Use this file only as a compact directory map.
