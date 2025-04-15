import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import os
import shutil

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Program Recommendation (Training and Validation)")
st.header("Training")

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("dataset/sample_dataset/train_with_summary.csv")

# ---------------------------
# 2. Load Embedding Model
# ---------------------------
model_name = "thenlper/gte-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# ---------------------------
# 3. Setup Chroma DB
# ---------------------------
persist_directory = "chroma_db_new"
collection_name = "SPM"

# Clear existing database if any
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# ---------------------------
# 4. Prepare Documents
# ---------------------------
documents = [
    Document(
        page_content=row["Summary"],
        metadata={
            "id": i + 1,
            "Pre_Uni_Program": row["Pre_Uni_Program"],
            "Uni_Program": row["Uni_Program"]
        }
    )
    for i, row in df.iterrows()
]

# ---------------------------
# 5. Create Chroma Collection
# ---------------------------
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=persist_directory,
    collection_name=collection_name
)
vectorstore.persist()
st.write("Chroma collection created and data inserted.")