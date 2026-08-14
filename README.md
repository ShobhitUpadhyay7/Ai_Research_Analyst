# AI Research Analyst

AI Research Analyst is a Dockerized LangChain + ChromaDB research assistant that searches internal documents, technical documentation, and web sources to produce a grounded, cited research report.

The system uses hybrid retrieval with BM25 and vector search, fuses results using Reciprocal Rank Fusion, reranks evidence, compresses context, generates a report, verifies citations, and detects conflicts between sources.

## Architecture

```text
USER
│
▼
AI AGENT
│
┌─────────────┼─────────────┐
▼             ▼             ▼
RAG Tool      Web Tool      Utility
│             │             │
└─────────────┼─────────────┘
▼
QUERY ROUTER
│
▼
QUERY TRANSFORMATION
│
┌────────┴────────┐
▼                 ▼
BM25           VECTOR SEARCH
│                 │
└────────┬────────┘
▼
RRF FUSION
│
▼
RERANKER
│
▼
CONTEXT COMPRESSION
│
▼
LLM / GENERATOR
│
▼
CITATION VERIFICATION
│
▼
FINAL REPORT