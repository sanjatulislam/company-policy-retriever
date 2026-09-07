import json

import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.retriever import retrieve_rag_contexts, rerank_documents
from scripts.llm_caller import get_response
from utilities.utils import format_context, get_rag_prompt
from scripts.query_decomposer import get_decomposed_queries
from utilities.constants import RAG_FALLBACK_RESPONSE, COHERE_TOP_N, RETRIEVER_TOP_K, MAX_RAG_RESPONSE_GENERATION_RETRIES


def generate_final_response(query, 
                            should_print_subqueries=False, 
                            should_print_ranking_score=False, 
                            reranking_top_n=COHERE_TOP_N):
    raw_sub_queries = get_decomposed_queries(query)
    
    if should_print_subqueries:
        print(raw_sub_queries)

    sub_queries = json.loads(raw_sub_queries)
    docs = retrieve_rag_contexts(sub_queries, should_print_ranking_score, reranking_top_n)
    context = format_context(docs)

    prompt_value = get_rag_prompt(query, context, RAG_FALLBACK_RESPONSE)
    response = get_response(prompt_value)

    return response.content


def generate_rag_response(query, 
                          should_print_subqueries=False, 
                          should_print_ranking_score=False, 
                          reranking_top_n=COHERE_TOP_N):
    for attempt in range(MAX_RAG_RESPONSE_GENERATION_RETRIES + 1):

        response = generate_final_response(
            query,
            should_print_subqueries=should_print_subqueries,
            should_print_ranking_score=should_print_ranking_score,
            reranking_top_n=reranking_top_n
        )

        if response.strip() != RAG_FALLBACK_RESPONSE:
            return response

        if attempt < MAX_RAG_RESPONSE_GENERATION_RETRIES:
            reranking_top_n = RETRIEVER_TOP_K

    return response
