import pandas as pd
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st

st.title("Program Recommendation (Training and Validation)")
st.header("Training")
# ---------------------------
# 1. Load Dataset
# ---------------------------
# Load Sample Training Dataset
df = pd.read_csv("dataset/sample_dataset/train.csv")

# ---------------------------
# 2. Initialize Milvus Client
# ---------------------------
# Connecting to a local Milvus vector DB (SQLite-based via milvus-lite or similar).
collection_name = "SPM"
client = MilvusClient('./my_milvus.db')

# ---------------------------
# 3. Load Embedding Model
# ---------------------------
# Initializing HuggingFace model to generate text embeddings.
# Using CPU and cosine-normalized embeddings.
model_name = "thenlper/gte-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# ---------------------------
# 4. Embed Text into Vectors
# ---------------------------
# Generate dense vector embeddings from the text data.
vectors = embedding_model.embed_documents(df["Pre_Uni_Results"].tolist())

# ---------------------------
# 5. Create Milvus Collection
# ---------------------------
# Drop the collection if it already exists to ensure a clean setup.
if collection_name in client.list_collections():
    client.drop_collection(collection_name)

# Create a new collection with the appropriate vector dimension.
vector_dim = len(vectors[0])
client.create_collection(
    collection_name=collection_name,
    dimension=vector_dim
)
st.write(f"Created new collection: {collection_name}")

# ---------------------------
# 6. Insert Data into Milvus
# ---------------------------
# Prepare data to insert: assign IDs explicitly and include text and vector.
insert_data = [
    {"id": i + 1, **df.iloc[i].to_dict(), "vector": vec}  # Include all columns from the DataFrame and embed the vector
    for i, (vec) in enumerate(vectors)
]

# Insert the prepared data into the collection.
client.insert(
    collection_name=collection_name, 
    data=insert_data
)
st.write("Data inserted into Milvus collection.")

# ---------------------------
# 7. Search with New Text
# ---------------------------
# Embed a new query text into a vector for similarity search.
st.header("Validation")
new_text = "The student achieved A in Bahasa Melayu. The student achieved A in English. The student achieved A+ in Mathematics. The student achieved A in Additional Mathematics. The student achieved A in Physics. The student achieved A in Chemistry. The student achieved A in Biology. The student achieved A in History. The student achieved A in Moral. The student achieved A in Accounting."
query_vector = embedding_model.embed_query(new_text)
st.write(f"\n**New Text:** {new_text}")

# Define search parameters using cosine similarity.
search_params = {
    "metric_type": "COSINE",
    "params": {}
}

# Perform the vector similarity search and return top 5 results.
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=5,
    output_fields=["Pre_Uni_Program","Pre_Uni_Results","Uni_Program"],
    search_params=search_params
)

# ---------------------------
# 8. Display Search Results 
# ---------------------------
st.header("Top 5 similar results:")
for i, user in enumerate(results[0]):
    st.write("\n")
    st.write(f"**Rank {i+1}:**")
    st.write(f"**id:** {user['id']}")
    st.write(f"**distance:** {user['distance']}")    
    if 'entity' in user and user['entity']:  # If entity has data
        for field, value in user['entity'].items():
            st.write(f"**{field}:** {value}")
    else:
        st.write("No entity data available.")
