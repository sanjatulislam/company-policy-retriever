from dotenv import load_dotenv

import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utilities.utils import get_decomposition_prompt
from scripts.llm_caller import get_response


load_dotenv()


def get_decomposed_queries(query):
    prompt_value = get_decomposition_prompt(query)
    response = get_response(prompt_value)

    return response.content