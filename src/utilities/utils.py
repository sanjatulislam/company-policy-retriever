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


def get_rag_prompt(query, context):
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
    
    
def get_decomposition_prompt(query, context):
    template = ChatPromptTemplate([
        (
            "system",
            """You are a query decomposition assistant for Suffolk County Council's policy assistant.

Your task is to analyze the user's question and break it into simpler, standalone sub-questions that can each be answered independently by searching policy documents.

You are provided with context retrieved from the knowledge base. Use this context to understand the topics, concepts, terminology, and
information available in the knowledge base.

Follow these rules:
1. For a simple single-topic question, return it unchanged as the only item in the array.
2. For a multi-topic question, split it into separate self-contained sub-questions - one per topic.
3. For comparisons or relationships between multiple topics, decompose it into ONE standalone sub-question per thing being related. Comparison itself happens after retrieval.
4. For broad or general questions, use the context to identify relevant aspects and create focused queries.
4. Preserve important terminology, entities and concepts from the user's question and the retrieved context.
5. Each sub-query must be suitable for semantic retrieval.
6. Respond with exactly one line: a plain JSON array of strings, starting with [ and ending with ]. This single line is your entire response.

Context:
{context}"""
        ),
        (
            "human",
            "{query}"
        )
    ])

    prompt_value = template.invoke({
        'query': query,
        'context': context
    })

    return prompt_value
