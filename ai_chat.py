import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Set page config
st.set_page_config(page_title="LangChain QA Bot", page_icon="💬", layout="centered")

# Title
st.title("💬 Ask your Question")
st.caption("Powered by LangChain, Ollama, and ChromaDB")

# Initialize Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# LLM setup
llm = ChatOllama(model="llama3.2", temperature=0)

# Vector DB
vector_store = Chroma(
    collection_name="SPM",
    embedding_function=embedding,
    persist_directory="chroma_db_new",
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question based only on the data provided."),
    ("human", "Use the user question {input} to answer the question. Use only the {context} to answer the question.")
])

# Chain setup
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Streamlit form
with st.form(key="qa_form"):
    user_question = st.text_input("Enter your question:", placeholder="Type your question here...")
    submit_button = st.form_submit_button(label="🔍 Ask")

if submit_button and user_question.strip() != "":
    with st.spinner("Thinking..."):
        response = retrieval_chain.invoke({"input": user_question})
        st.markdown("### 📌 Answer:")
        st.success(response['answer'])
