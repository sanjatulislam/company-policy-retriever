import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.retriever import retrieve_initial_context
from utilities.utils import get_decomposition_prompt
from scripts.llm_caller import get_response


def get_decomposed_queries(query):
    context = retrieve_initial_context(query)
    prompt_value = get_decomposition_prompt(query, context)
    response = get_response(prompt_value)

    return response.content