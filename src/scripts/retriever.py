import os
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth 
from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.utilities.constants import EMBEDDING_MODEL, WEAVIATE_TEXT_KEY, TOP_K

load_dotenv()

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
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
        search_kwargs={"k": TOP_K}
    )

    return client, retriever


def retrieve(query):
    weaviate_client, retriever = get_retriever()

    results = retriever.invoke(query)

    weaviate_client.close()

    return results

    