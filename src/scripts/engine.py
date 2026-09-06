import os 
from dotenv import load_dotenv

import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.retriever import retrieve_query, retrieve_queries
from scripts.llm_caller import get_response
from utilities.utils import format_context, get_chat_prompt
from scripts.query_decomposer import get_decomposed_queries


def generate_final_response(query):
    docs = retrieve_query(query)
    context = format_context(docs)
    prompt_value = get_chat_prompt(query, context)

    response = get_response(prompt_value)

    return response.content


def generate_final_response_v2(query):
    sub_queries = get_decomposed_queries(query)
    docs = retrieve_queries(query, sub_queries)
    context = format_context(docs)
    prompt_value = get_chat_prompt(query, context)

    response = get_response(prompt_value)

    return response.content
