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
    
    
