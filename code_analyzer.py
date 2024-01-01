import os
from api_key import *
from langchain.document_loaders import DirectoryLoader, PythonLoader
from langchain.text_splitter import (
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
FAISS_DB = os.path.join(path, "faiss_db", "code_index")
CODE_PATH = os.path.join(path, "data", "code")

# initialize embedder
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hugging_face_key,
    model_name="sentence-transformers/all-MiniLM-l6-v2",
)


vectordb = None
if not os.path.exists(FAISS_DB):
    print("loading from local code and store in local db")
    loader = DirectoryLoader(CODE_PATH, glob="**/*.py", loader_cls=PythonLoader)
    code = loader.load()

    # splitting documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = text_splitter.split_documents(code)

    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectordb.save_local(FAISS_DB)

if vectordb is None:
    vectordb = FAISS.load_local(FAISS_DB, embeddings)


retriever = vectordb.as_retriever()

# define llm model
llm = GooglePalm(google_api_key=api_key)
# temperature will less when we are asking question on a given context.
llm.temperature = 0.4

# question = "Find the chapter names and list them"

# practice
question = "create 2 questions from it."
# Your task is analyze the give codebase and give feedback as presice as possible.
prompt_template = """
Your are a python expert. 
Your task is create some questions from the given context.

DON'T MAKE THING UP. 
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
