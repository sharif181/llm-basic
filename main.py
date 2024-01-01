from api_key import api_key, hugging_face_key
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceInstructEmbeddings,
    GooglePalmEmbeddings,
)
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate

# from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from langchain.vectorstores import FAISS

inference_api_key = hugging_face_key

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key,
    model_name="sentence-transformers/all-MiniLM-l6-v2",
)

# embeddings = GooglePalmEmbeddings(google_api_key=api_key)


# text_splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False,
# )

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
FAISS_INDEX_PATH = os.path.join(path, "faiss_db", "faiss_index")

vectordb = None

if not os.path.exists(FAISS_INDEX_PATH):
    print("loading from url and store in local")
    # data source
    # loader = WebBaseLoader("https://www.cricbuzz.com/")
    loader = WebBaseLoader(
        "https://www.cricbuzz.com/cricket-match-squads/82717/indw-vs-ausw-only-test-australia-women-tour-of-india-2023-24"
    )
    # loaded data
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local("faiss_index")

# # embeddings = HuggingFaceEmbeddings()
# # embeddings = HuggingFaceInstructEmbeddings(
# #     query_instruction="Represent the query for retrieval: "
# # )


# # e = embeddings.embed_query("who win today?")
# # print(e[:5])

if vectordb is None:
    vectordb = FAISS.load_local("faiss_index", embeddings)

# query = "What was india's playing eleven? also tell me what is there role in team?"
query = "Is there anything related to richa ghosh?"
# # embedding_vector = embeddings.embed_query(query)
# # docs = vectordb.similarity_search_by_vector(embedding_vector)

# # print(docs[0])

llm = GooglePalm(google_api_key=api_key)
# temperature will less when we are asking question on a given context.
llm.temperature = 0.2

retriever = vectordb.as_retriever()
# print(retriever.get_relevant_documents(query))


# prompt template
prompt_template = """
Try to find answer from the given context. Don't try to create answer yourself.
Look into the given context and provide answer. If answer is not in the context say "I don't know".

return question: question text. then answer in a newline with source from context.

CONTEXT: {context}

QUESTION: {question}
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    input_key="query",
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt},
)

print(chain(query))
