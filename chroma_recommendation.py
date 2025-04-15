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
df = pd.read_csv("dataset/sample_dataset/train.csv")

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
persist_directory = "chroma_db"
collection_name = "SPM"

# Clear existing database if any
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# ---------------------------
# 4. Prepare Documents
# ---------------------------
documents = [
    Document(
        page_content=row["Pre_Uni_Results"],
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

# ---------------------------
# 6. Query New Text
# ---------------------------
st.header("Validation")
new_text = "The student achieved A in Bahasa Melayu. The student achieved A in English. The student achieved A+ in Mathematics. The student achieved A in Additional Mathematics. The student achieved A in Physics. The student achieved A in Chemistry. The student achieved A in Biology. The student achieved A in History. The student achieved A in Moral. The student achieved A in Accounting."
st.write(f"\n**New Text:** {new_text}")

# Search
results = vectorstore.similarity_search_with_score(new_text, k=5)

# ---------------------------
# 7. Display Results
# ---------------------------
st.header("Top 5 similar results:")
for i, (doc, score) in enumerate(results):
    st.write("\n")
    st.write(f"**Rank {i+1}:**")
    st.write(f"**Similarity Score (lower is better):** {score:.4f}")
    st.write(f"**Pre_Uni_Results:** {doc.page_content}")
    for key, value in doc.metadata.items():
        st.write(f"**{key}:** {value}")