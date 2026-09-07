import os 
from dotenv import load_dotenv

from langchain_groq import ChatGroq

import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utilities.constants import GENERATION_LLM, GENERATION_TEMPERATURE, GENERATION_LLM_MAX_TOKENS


load_dotenv()

llm = ChatGroq(
        model=GENERATION_LLM,
        api_key=os.environ['GROQ_API_KEY'],
        max_tokens=GENERATION_LLM_MAX_TOKENS,
        temperature=GENERATION_TEMPERATURE
    )

def get_response(prompt_value):
    response = llm.invoke(prompt_value)
    return response