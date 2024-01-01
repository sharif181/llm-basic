import os
from api_key import *
from langchain.chat_models import ChatGooglePalm
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
)
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

# from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.memory import ConversationSummaryMemory

# code start

# CONSTANT VALUES
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# file path initializations
path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
FAISS_DB = os.path.join(path, "faiss_db", "pdf_index")
book_path = os.path.join(path, "data", "books", "ml.pdf")

if not os.path.exists(book_path):
    print("book not exists")
    exit()

# initialize embedder
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hugging_face_key,
    model_name="sentence-transformers/all-MiniLM-l6-v2",
)


vectordb = None
if not os.path.exists(FAISS_DB):
    print("loading from pdf and store in local db")
    loader = PyPDFLoader(book_path)
    pages = loader.load()

    # splitting documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = text_splitter.split_documents(pages)

    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectordb.save_local(FAISS_DB)

if vectordb is None:
    vectordb = FAISS.load_local(FAISS_DB, embeddings)


llm = ChatGooglePalm(google_api_key=api_key, temperature=0.5)
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)
retriever = vectordb.as_retriever()
while True:
    question = input("Enter your question: ")
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    res = qa(question)
    print(res.get("answer"))
