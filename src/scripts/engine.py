import os 
from dotenv import load_dotenv

import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.retriever import retrieve
from scripts.llm_caller import get_llm_response
from utilities.utils import format_context, get_chat_prompt


def generate_final_response(query):
    docs = retrieve(query)
    context = format_context(docs)
    prompt_value = get_chat_prompt(query, context)

    response = get_llm_response(prompt_value)

    return response.content
