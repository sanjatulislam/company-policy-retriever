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


def get_rerank_retriever(retriever):
    compressor = CohereRerank(cohere_api_key=os.environ['COHERE_API_KEY'], 
                              model=COHERE_RERANK_MODEL, 
                              top_n=COHERE_TOP_N)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=retriever
    )

    return compression_retriever


def retrieve(query):
    # TODO: Implement query analyzer/decomposer to split multi-topic questions
    # into focused sub-queries for retrieval

    weaviate_client, retriever = get_retriever()
    cohere_retriever = get_rerank_retriever(retriever)

    results = cohere_retriever.invoke(query)

    weaviate_client.close()

    return results

    