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

ollow these rules:
1. If the question is simple, single-topic, and asks about one policy concept, return it unchanged as the only item in the array.
2. If the question covers multiple distinct topics, split it into separate self-contained sub-questions — one per topic.
3. If the question asks how, whether, or why TWO OR MORE things relate, compare, connect, differ, or conflict, decompose it into ONE standalone sub-question per thing being related. Phrase each sub-question about a single document or concept; the comparison itself happens after retrieval, not during it.
4. Keep each sub-question clear and answerable on its own, without relying on the original question or other sub-questions for context.
5. Respond with exactly one line: a plain JSON array of strings, starting with [ and ending with ]. This single line is your entire response.

Example 1:
Question: "What is the annual leave policy?"
Output: ["What is the annual leave policy?"]

Example 2:
Question: "What are the policies for annual leave, sick leave, and parental leave?"
Output: ["What is the annual leave policy?", "What is the sick leave policy?", "What is the parental leave policy?"]

Example 3:
Question: "How does the Information Security Policy's risk assessment process (DPIA) relate to the DPIA requirement mentioned for information sharing in the Data Protection Policy?"
Output: ["What is the DPIA (data protection impact assessment) risk assessment process described in the Information Security Policy?", "What does the Data Protection Policy say about DPIA requirements for information sharing?"]
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
