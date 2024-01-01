import os
from api_key import *
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    TokenTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from langchain.vectorstores import FAISS

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

# defined retriever
# retriever = vectordb.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 5,
#         "fetch_k": 20,
#         # "filter": {"paper_title": "GPT-4 Technical Report"},
#     },
# )

retriever = vectordb.as_retriever()

# define llm model
llm = GooglePalm(google_api_key=api_key)
# temperature will less when we are asking question on a given context.
llm.temperature = 0.4

# question = "Find the chapter names and list them"

# practice
question = "what is machine learning? with example"

prompt_template = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, 
don't try to make up an answer. 
ALWAYS use words from the context. 
CONTEXT: {context}
QUESTION: {question}
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

print(qa_chain(question).get("result"))
# chapters = qa_chain(question).get("result").split("\n")

# questions = []

# for chapter in chapters:
#     context = """
#             from {} chapter.
#             Generate 2 MCQ questions with 4 options from this given context.
#             Always follow provided format.
#             Question Format will be:
#             Q1. **Question will be here**
#             A. option1
#             B. option2
#             C. option3
#             D. option4

#             Also provide corrent answer in a seperate line.
#             Answer format will be:
#             Answer: (A)
#         """.format(
#         chapter
#     )
#     # print(context)
#     print(qa_chain(context).get("result"))
