from api_key import api_key, hugging_face_key, figma_access_token
from langchain.document_loaders.figma import FigmaFileLoader
from urllib.parse import urlparse, parse_qs
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
)
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)


# code start
# CONSTANT VALUES
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

FAISS_INDEX_PATH = os.path.join(path, "faiss_db", "figma_context")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hugging_face_key,
    model_name="sentence-transformers/all-MiniLM-l6-v2",
)


def extract_key_and_node_id(figma_url):
    parsed_url = urlparse(figma_url)

    # Extract key from the path
    path_parts = parsed_url.path.split("/")
    key_index = path_parts.index("file") + 1 if "file" in path_parts else -1
    key = path_parts[key_index] if key_index != -1 else None

    # Extract node ID from query parameters
    query_params = parse_qs(parsed_url.query)
    node_id = query_params.get("node-id", [None])[0]

    return key, node_id


url = "https://www.figma.com/file/jXL178D4qkvBfbaBrL9A03/QuizifyPDF?type=design&node-id=0%3A1&mode=design&t=Xq2yTMGzXtQ804OI-1"


# Extract key and node ID
key, node_id = extract_key_and_node_id(url)

# if key and node_id:
#     print(f"Key: {key}")
#     print(f"Node ID: {node_id}")
# else:
#     print("Invalid Figma URL or structure.")

vectordb = None
if not os.path.exists(FAISS_INDEX_PATH):
    print("loading from figma and store in faiss")
    figma_loader = FigmaFileLoader(
        access_token=figma_access_token,
        ids=node_id,
        key=key,
    )
    documents = figma_loader.load()
    # splitting documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = text_splitter.split_documents(documents)
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)


if vectordb is None:
    print("faiss loading from local")
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings)


# prompt template
prompt_template = """
Your are a UI/UX expert. Your task is analyze the given figma context and Answer following questinons.
based on the given context. DON'T make things up. Just answer the question as precise as possible.

CONTEXT: {context}
Question: {question}
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

query = (
    "Can you generate 2 questions from the figma design? and provide possible answer"
)
print(chain(query).get("result"))
