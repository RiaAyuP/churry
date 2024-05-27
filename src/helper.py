from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# load documents
def load_pdf(data):
    loader = DirectoryLoader(data, glob=".pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# split documents
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# define embedding
def embedding_in_use():
    embeddings = OpenAIEmbeddings()
    return embeddings
