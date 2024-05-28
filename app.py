from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf, text_split, embedding_in_use
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

vectordb = Chroma(
    persist_directory = "./chroma",
    embedding_function = OpenAIEmbeddings(),
)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa=RetrievalQA.from_chain_type(
    llm = llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT})

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)




