### Company Policy RAG Assistant

An advanced Retrieval-Augmented Generation (RAG) system for answering questions about company policies using retrieved policy documents as the source of truth.

The system combines context-aware query decomposition, hybrid retrieval, document reranking, deduplication and grounded LLM generation to improve retrieval relevance and reduce unsupported responses.

## RAG pipeline

**Data Ingestion Pipeline**

                    Policy Documents
                          │
                          ▼
                    Document Loading
                          │
                          ▼
                    Text Splitting
                          │
                          ▼
                    Document Chunks
                          │
                          ▼
                    Embedding Generation
                          │
                          ▼
                    Vector + Metadata
                          │
                          ▼
                       Weaviate

**Retrieval Pipeline**

                    User Query
                        │
                        ▼
              Initial Retrieval
                        │
                        ▼
          Context-Aware Query Decomposition
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
           Subquery   Subquery   Subquery
              │         │         │
              ▼         ▼         ▼
          Retrieval  Retrieval  Retrieval
              │         │         │
              ▼         ▼         ▼
           Rerank    Rerank    Rerank
              │         │         │
              └─────────┼─────────┘
                        ▼
                  Deduplication
                        │
                        ▼
                 Prompt Augmentation
                        │
                        ▼
                       LLM
                        │
                        ▼
                  Final Answer



## Key Features
- **Context-aware query decomposition**: Uses an initial retrieval result to identify multi-topic questions and decompose them into standalone search queries.
- **Semantic retrieval**: Used semantic search to retrieve data
- **Multi-query retrieval**: Retrieves evidence independently for each generated sub-query.
- **Document reranking**: Uses Cohere Rerank to reorder retrieved documents based on their relevance to the query.
- **Document deduplication**: Removes duplicate documents retrieved across multiple sub-queries.
- **Grounded generation**: The final LLM is instructed to answer strictly from the retrieved policy context.
- **Fallback and retry mechanism**: Tested using single-fact questions, cross-document questions, chunk-boundary cases and out-of-scope questions. Includes both manual evaluation and LLM-as-a-judge evaluation.


## Evaluation
- The system was evaluated using Single-Fact Retrieval, Cross-Document Synthesis, Chunk-Boundary Stress Tests and Out-of-Scope Questions
- The results were assessed using: Manual evaluation and LLM-as-a-judge evaluation
- he evaluation focuses on aspects such as: Retrieval relevance, Answer correctness, Context grounding, Handling of unsupported questions, Cross-document reasoning

## Technology Stack
Python, LangChain, Weaviate, Hugging Face Embeddings, Semantic Retrieval, Cohere Rerank, Groq
