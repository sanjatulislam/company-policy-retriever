import json

import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.retriever import retrieve_rag_contexts, rerank_documents
from scripts.llm_caller import get_response
from utilities.utils import format_context, get_rag_prompt
from scripts.query_decomposer import get_decomposed_queries


def generate_rag_response(query, should_print_subqueries=False):
    # TODO: Implement fallback mechanism

    raw_sub_queries = get_decomposed_queries(query)

    if should_print_subqueries:
        print(raw_sub_queries)

    sub_queries = json.loads(raw_sub_queries)
    docs = retrieve_rag_contexts(sub_queries)
    context = format_context(docs)

    prompt_value = get_rag_prompt(query, context)
    response = get_response(prompt_value)

    return response.content
