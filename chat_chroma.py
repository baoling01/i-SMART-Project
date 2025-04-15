import streamlit as st
from PIL import Image
import PyPDF2
import pytesseract
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="Program Recommendation", page_icon="https://4.bp.blogspot.com/-_EMlBTSVU6E/Tha_OAK8nMI/AAAAAAAAACY/azBH7qFTRlg/s1600/mmu-logo_m.jpg", layout="wide")
st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://cdforum.lne.st/wp-content/uploads/sites/61/2022/03/MMU-New-Secondary-logo.png" alt="MMU Logo" style="width: 120px; margin-right: 20px;">
        <h1 style="margin: 0;">MMU Program Recommendation System</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Powered by LangChain, Ollama, and ChromaDB")

# -------------------- Environment Setup --------------------
pytesseract.pytesseract.tesseract_cmd = r"/mnt/c/Program Files/Tesseract-OCR/tesseract.exe"
os.environ['PATH'] = r'/mnt/c/poppler/poppler-24.08.0/Library/bin' + ";" + os.environ['PATH']

# -------------------- Constants --------------------
N = 3  # Number of recommended programs
K = 10

# -------------------- Embedding + VectorDB + LLM --------------------
embedding = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

llm = ChatOllama(model="llama3.2", temperature=0)

vector_store = Chroma(
    collection_name="SPM",
    embedding_function=embedding,
    persist_directory="chroma_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k": K})

# -------------------- File Processing --------------------
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return "".join(page.extract_text() for page in pdf_reader.pages)
    except:
        return None

def extract_text_from_image(uploaded_file):
    img = Image.open(uploaded_file)
    return pytesseract.image_to_string(img)

# -------------------- User Input --------------------
def user_input():
    pre_university_programs = [
        "SPM (Sijil Pelajaran Malaysia)",
        "STPM (Sijil Tinggi Persekolahan Malaysia)",
        "IGCSE (International General Certificate of Secondary Education)",
        "A-Levels",
        "Foundation Program",
        "Matriculation",
        "UEC (Unified Examination Certificate)"
    ]

    selected_program = st.selectbox("Select your pre-university program:", pre_university_programs)
    uploaded_file = st.file_uploader("Upload your result document (PDF)", type=["pdf"])

    extracted_text = ""
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type in ["image/jpeg", "image/png"]:
            extracted_text = extract_text_from_image(uploaded_file)
        else:
            st.error("Unsupported file type.")
    return selected_program, extracted_text

# -------------------- Prompt Construction --------------------
def get_prompt(selected_program, extracted_text, similar_docs):
    formatted_similar_users = ""
    for i, doc in enumerate(similar_docs):
        formatted_similar_users += f"**Rank {i+1}**  \n"
        
        # Display all metadata fields
        for key, value in doc.metadata.items():
            formatted_similar_users += f"**{key}:** {value}  \n"
        
        formatted_similar_users += f"**Pre-University Results:** {doc.page_content}  \n\n"

    return f"""
    Given:
    1. Pre-University Program (User): {selected_program}
    2. Pre-University Result (User): {extracted_text}
    3. Top-N Similar Users:
    {formatted_similar_users}

    Task: 
    Recommend Top-{N} most suitable MMU (Multimedia University) undergraduate programs for this user.

    Your output should be in the following format:

    **Program Recommendations:**
    1. **<Program Name 1>:** <Reason>
    2. **<Program Name 2>:** <Reason>
    3. **<Program Name 3>:** <Reason>
    """

# -------------------- Main Recommendation Logic --------------------
def get_program_recommendation(selected_program, extracted_text):
    similar_docs = retriever.get_relevant_documents(extracted_text)
    prompt_input = get_prompt(selected_program, extracted_text, similar_docs)
    
    # Use Langchain with ChatOllama and prompt template
    llm = ChatOllama(model="llama3.2", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based only on the data provided."),
        ("human", "{input}")
    ])
    chain = prompt | llm
    
    result = chain.invoke({"input": prompt_input})        
    st.write(f"### Top-{N} Program Recommendation:")
    st.markdown(result.content)

# -------------------- Streamlit Main --------------------
def main():
    selected_program, extracted_text = user_input()
    if extracted_text:
        if st.button("🔍 Get Program Recommendations"):
            with st.spinner("Thinking..."):
                get_program_recommendation(selected_program, extracted_text)

if __name__ == "__main__":
    main()
