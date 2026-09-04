from dotenv import load_dotenv
import os

import weaviate
from weaviate.classes.init import Auth

from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utilities.utils import get_data_dir_path
from utilities.constants import WEAVIATE_TEXT_KEY

load_dotenv()

def load_documents():
    documents = []
    data_dir = get_data_dir_path()

    for pdf in data_dir.glob("*.pdf"):
        loader = PyPDFLoader(file_path=pdf)
        docs = loader.load()

        documents.extend(docs)

    print(f"Loaded {len(documents)} pages")

    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    return chunks



def update_chunk_metadata(chunks):

    all_chunks = []

    if chunks:
        for i, chunk in enumerate(chunks):

            page_label = chunk.metadata.get("page_label") or ""
            source = chunk.metadata.get("source")

            doc_name = source.split("\\")[-1]
            doc_name_pref = doc_name.split(".")[0]
            doc_name_ext = doc_name.split(".")[-1]

            doc = Document(page_content=chunk.page_content, metadata={"source": doc_name, "chunk_id": f"{doc_name_pref}_{i+1}.{doc_name_ext}", "page_label": page_label})

            all_chunks.append(doc)

    print(f"Total chunks: {len(all_chunks)}")

    return all_chunks




def store_documents(chunks, should_delete_previous_data=True, embedding_model="BAAI/bge-m3"):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True}
    )

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url = os.environ['WEAVIATE_URL'],
        auth_credentials=Auth.api_key(os.environ['WEAVIATE_ADMIN_API_KEY'])
    )

    if should_delete_previous_data:
        client.collections.delete_all()

    store = WeaviateVectorStore(
        client=client,
        index_name=os.environ['WEAVIATE_COLLECTION'],
        text_key=WEAVIATE_TEXT_KEY,
        embedding=embeddings
    )

    store.add_documents(chunks)

    print(f"Stored {len(chunks)} vector database")

    client.close()


def run_ingestion_pipeline(should_delete_previous_data=True, embedding_model="BAAI/bge-m3"):
    documents = load_documents()
    chunks = chunk_documents(documents)
    chunks = update_chunk_metadata(chunks)
    store_documents(chunks, should_delete_previous_data=should_delete_previous_data, embedding_model=embedding_model)


if __name__ == "__main__":
    run_ingestion_pipeline()   



