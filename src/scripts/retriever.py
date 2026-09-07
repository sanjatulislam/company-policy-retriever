import os
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth 
from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utilities.constants import EMBEDDING_MODEL, WEAVIATE_TEXT_KEY, RETRIEVER_TOP_K, EMBEDDING_MODEL_QUERY_INSTRUCTION, COHERE_RERANK_MODEL, COHERE_TOP_N

load_dotenv()

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={
            "prompt": EMBEDDING_MODEL_QUERY_INSTRUCTION,
            "normalize_embeddings": True,
        },
    )

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ['WEAVIATE_URL'],
        auth_credentials=Auth.api_key(os.environ['WEAVIATE_VIEWER_API_KEY'])
    )

    vector_db = WeaviateVectorStore(
        client=client,
        index_name=os.environ['WEAVIATE_COLLECTION'],
        text_key=WEAVIATE_TEXT_KEY,
        embedding=embeddings
    )

    retriever = vector_db.as_retriever(
        search_kwargs={"k": RETRIEVER_TOP_K}
    )

    return client, retriever


def get_rerank_compressor(reranking_top_n=COHERE_TOP_N):
    compressor = CohereRerank(cohere_api_key=os.environ['COHERE_API_KEY'], 
                              model=COHERE_RERANK_MODEL, 
                              top_n=reranking_top_n)

    return compressor


def get_rerank_compressor_retriever(retriever):
    compressor = get_rerank_compressor()

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=retriever
    )

    return compression_retriever


def retrieve_initial_context(query):
    weaviate_client, retriever = get_retriever()

    try:
        response = retriever.invoke(query)

        return response

    finally:
        weaviate_client.close()


def deduplicate(documents):
    seen = set()
    unique_docs = []
    
    for doc in documents:
        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id is None:
            unique_docs.append(doc)
            continue

        if chunk_id not in seen:
            unique_docs.append(doc)
            seen.add(chunk_id)

    return unique_docs


def retrieve_rag_contexts(sub_queries, 
                          should_print_ranking_score=False, 
                          reranking_top_n=COHERE_TOP_N):
    weaviate_client, retriever = get_retriever()
    all_docs = []

    try:
        for query in sub_queries:
            docs = retriever.invoke(query)
            ranked_docs = rerank_documents(query, docs, should_print_ranking_score, reranking_top_n)
            all_docs.extend(ranked_docs)

        unique_docs = deduplicate(all_docs)

        return unique_docs

    finally:
        weaviate_client.close()


def rerank_documents(query, 
                     documents, 
                     should_print_score=False, 
                     reranking_top_n=COHERE_TOP_N):
    compressor = get_rerank_compressor(reranking_top_n)
    docs = compressor.compress_documents(documents=documents, query=query)

    if should_print_score:
        print(f"\n--- Reranked results ({len(docs)}) ---")
        for i, doc in enumerate(docs):
            score = doc.metadata.get('relevance_score', 'N/A')
            print(f"{i+1}. [{score}] {doc.metadata.get('chunk_id')}: {doc.page_content[:100]}...")

    return docs
