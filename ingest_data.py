"""Python file to serve as the frontend"""
import os
import pickle
from dotenv import load_dotenv
import openai


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Data
loader = UnstructuredFileLoader("boiler.txt")
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)


# Load Data to vectorstore
embeddings = OpenAIEmbeddings()
my_vectorstore = FAISS.from_documents(documents, embeddings)

# Save vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(my_vectorstore, f)