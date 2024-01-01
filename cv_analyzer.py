from api_key import api_key, hugging_face_key
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
)
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm


# code
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hugging_face_key,
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    # model_name="hkunlp/instructor-large",
)

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

FAISS_INDEX_PATH = os.path.join(path, "faiss_db", "job_context")

vectordb = None
if not os.path.exists(FAISS_INDEX_PATH):
    print("loading from local pdf and store in faiss")
    file_path = os.path.join(path, "data", "demo_context_3.pdf")

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    vectordb = FAISS.from_documents(documents=pages, embedding=embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)


if vectordb is None:
    print("faiss loading from local")
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings)


file_path = os.path.join(path, "data", "test.pdf")
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

query = ""
for page in pages:
    query += page.page_content


# prompt template
prompt_template = """
You will be given a job description as context. Also a CV of a candidate.
Your task will be select that candidate or reject based on the given context and CV.
You will act like a hiring manager. 

**Alwasy return valid reason for rejaction or acception**

CONTEXT: {context}
CV: {question}
"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

llm = GooglePalm(google_api_key=api_key)
llm.temperature = 0.5

retriever = vectordb.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    input_key="query",
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt},
)


print(chain(query).get("result"))
