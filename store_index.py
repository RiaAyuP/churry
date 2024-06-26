from src.helper import load_pdf, text_split
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

extracted_data = load_pdf("data/")

text_chunks = text_split(extracted_data)

vectordb = Chroma.from_documents(documents = text_chunks, embedding = OpenAIEmbeddings(), persist_directory = "./chroma")
vectordb.persist()