from src.helper import load_pdf, text_split, embedding_in_use
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# initializing ChromaDB
if __name__ == "__main__":
    load_dotenv()
    if os.path.exists("./chroma"):
        print("already embedded")
        exit(0)
   
    extracted_data = load_pdf("data/")

    text_chunks = text_split(extracted_data)

    embeddings = embedding_in_use()

    vectordb = Chroma.from_documents(documents = text_chunks, embedding = embeddings, persist_directory = "./chroma")
    vectordb.persist()