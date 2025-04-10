# ---------------------------------------------------------------------------------------------------------
# Deployment (Testing)
# ---------------------------------------------------------------------------------------------------------
import streamlit as st
from PIL import Image
import PyPDF2
import pytesseract
import os
from pymilvus import MilvusClient
from langchain.embeddings import HuggingFaceEmbeddings
import ollama
import json

# Set path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"/mnt/c/Program Files/Tesseract-OCR/tesseract.exe"  # For WSL (Windows Subsystem for Linux)
os.environ['PATH'] = r'/mnt/c/poppler/poppler-24.08.0/Library/bin' + ";" + os.environ['PATH']

# Global Variables
N = 3
K = 10

# Function to extract text from text-based PDF
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        return None  # If extraction fails, return None

# Function to extract text from image-based PDFs using OCR
def extract_text_from_image_based_pdf(uploaded_file):
    # Convert PDF to images using PyMuPDF or pdf2image
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(uploaded_file, 500)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page)
        return text
    except Exception as e:
        return "Error during image-based PDF extraction."

# Function to extract text from image files (JPG, PNG)
def extract_text_from_image(uploaded_file):
    img = Image.open(uploaded_file)
    text = pytesseract.image_to_string(img)  # OCR using pytesseract
    return text

# Function to capture user inputs
def user_input():
    # Ask user to select pre-university programs
    pre_university_programs = [
        "SPM (Sijil Pelajaran Malaysia)",
        "STPM (Sijil Tinggi Persekolahan Malaysia)",
        "IGCSE (International General Certificate of Secondary Education)",
        "A-Levels",
        "Foundation Program",
        "Matriculation",
        "UEC (Unified Examination Certificate)"
    ]

    selected_programs = st.selectbox(
        "Select the pre-university programs you have taken:",
        pre_university_programs
    )

    # Ask user to upload their result document (PDF or image)
    uploaded_file = st.file_uploader("Upload your result document (PDF, JPG, or PNG)", type=["pdf", "jpg", "png"])

    extracted_text = ""
    if uploaded_file is not None:
        # Check if file is a PDF
        if uploaded_file.type == "application/pdf":
            # Attempt to extract text from text-based PDF
            extracted_text = extract_text_from_pdf(uploaded_file)
            if not extracted_text:  # If text extraction failed, use OCR for image-based PDF
                extracted_text = "Text extraction failed. Attempting OCR for image-based PDF...\n" + extract_text_from_image_based_pdf(uploaded_file)

        # Check if file is an image (JPG, PNG)
        elif uploaded_file.type in ["image/jpeg", "image/png"]:
            extracted_text = extract_text_from_image(uploaded_file)

        else:
            extracted_text = "Unsupported file type."

    return selected_programs, extracted_text

# Top-K Similar Users
def extract_top_k_similar_users(extracted_text, collection_name="SPM"):
    # Load Milvus Vector Database
    client = MilvusClient("./my_milvus.db")

    # Load Embedding Model
    model_name = "thenlper/gte-large"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Embed a new query text into a vector for similarity search
    query_vector = embedding_model.embed_query(extracted_text)

    # Define search parameters using cosine similarity.
    search_params = {
        "metric_type": "COSINE",
        "params": {}
    }

    # Perform the vector similarity search and return top 5 results.
    results = client.search(
        collection_name=collection_name, # This should be equivalent to pre-university results, which means each pre-university programs indicating one collection
        data=[query_vector],
        limit=K,
        output_fields=["Pre_Uni_Results","Uni_Program"], # Output field (Result and Program)
        search_params=search_params
    )

    return results

# Function to get LLM Prompt
def get_prompt(selected_program, extracted_text, formatted_similar_users):
    prompt = f"""
        Given:
        1. Pre-University Program (User): {selected_program}
        2. Pre-University Result (User): {extracted_text}
        3. Top-N Similar Users:
        {formatted_similar_users}

        Task: 
        Recommend Top-{N} most suitable MMU (Multimedia University) undergraduate program for this user.

        Your output should be in the following format:

        **Program Recommendations:**
        1. **<Program Name 1>:** <Reason why it fits the user's profile>
        2. **<Program Name 2>:** <Reason...>
        3. **<Program Name 3>:** <Reason...>
        ...
        {N}. **<Program Name {N}:** <Reason...>
    """
    return prompt

# Top-N Recommended Program
def get_program_recommendation(selected_program, extracted_text):
    # Extract Top-K Similar Users
    st.write(f"### Top-{K} Similar Historical Students")
    results = extract_top_k_similar_users(extracted_text) # put selected program later
    top_k_similar_users = results[0] 
    formatted_similar_users = ""
    for i, user in enumerate(top_k_similar_users):
        entity = ""
        if 'entity' in user and user['entity']:
            for field, value in user['entity'].items():
                entity += f"\n**{field}:** {value}\n"
        formatted_similar_users += f"""
            \n**Rank {i+1}**:\n
            \n**ID:** {user['id']}\n
            \n**Distance between user and existing data:** {user['distance']}\n
            \n{entity}\n
        """
    st.write(formatted_similar_users)

    # Top-N Program Recommendation
    st.write(f"### Top-{N} Program Recommendations:")
    prompt = get_prompt(selected_program, extracted_text, formatted_similar_users)
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    if isinstance(response, str):
        response = json.loads(response)
    recommendation_text = response["message"].get("content", "")
    st.write(recommendation_text)

# Streamlit application entry point
def main():
    st.title("MMU Program Recommendation")
    selected_programs, extracted_text = user_input()
    if extracted_text:
        st.write("### Extracted Text from Uploaded File:")
        st.write(extracted_text)
        get_program_recommendation(selected_programs, extracted_text)

# Run the application      
if __name__ == "__main__":
    main()

# In real application, only Top-N Program will be shown.
