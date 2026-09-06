from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

def get_data_dir_path(data_dir="data"):
    dir_path = ""
    curr_dir = Path(__file__).resolve()

    for parent in curr_dir.parents:
        if (parent / data_dir).exists():
            dir_path = parent / data_dir
            break

    return dir_path


def format_context(documents):
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in documents
    )

    return context


def get_chat_prompt(query, context):
    template = ChatPromptTemplate(
        [
            (
                "system",
                """You are Suffolk County Council's policy assistant. Your role is to answer questions strictly using the policy documents provided in the context below.
Read the context carefully and base your answer only on information explicitly stated there.
Keep the answer concise and factual. If the context only partially answers the question, share what is supported by the context and clearly state which parts are not covered.
If the context does not contain relevant information, respond exactly with: 'I could not find the answer in the provided documents.'

Context:
{context}"""
            ),
            (
                "human",
                "{query}"
            )
        ]
    )

    prompt_value = template.invoke({
        "context": context,
        "query": query
    })

    return prompt_value
    
    
def get_decomposition_prompt(query):
    template = ChatPromptTemplate([
        (
            "system",
            """You are a query decomposition assistant for Suffolk County Council's policy assistant.

Your task is to break down the user's question into simpler, standalone sub-questions that can each be answered independently by searching policy documents.

Follow these rules:
1. If the question is already simple and single-topic, return it unchanged as the only item in the array.
2. If the question covers multiple topics or asks multiple things, split it into separate, self-contained sub-questions.
3. Keep each sub-question clear and answerable on its own, without relying on the original question for context.
4. Return only a JSON array of strings, with no extra text, preamble or explanation.

Example 1:
Question: "What is the annual leave policy?"
Output: ["What is the annual leave policy?"]

Example 2:
Question: "What are the policies for annual leave, sick leave, and parental leave?"
Output: ["What is the annual leave policy?", "What is the sick leave policy?", "What is the parental leave policy?"]
"""
        ),
        (
            "human",
            "{query}"
        )
    ])

    prompt_value = template.invoke({
        'query': query
    })

    return prompt_value
