import os
import logging
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import AzureOpenAI
import numpy as np
import faiss

# Load credentials from .env file
load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Initializing Azure OpenAI Client.")
# Initialize Azure OpenAI Client
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2024-09-01-preview"
)


# Parse PDF Document
def parse_pdf(file):
    logger.info("Parsing PDF document.")
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text


# Chunk Text for Embedding
def chunk_text(text, max_chunk_size=300):
    logger.info("Chunking text into manageable pieces.")
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk + sentence) <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


# Generate Embeddings
def generate_embeddings(text_chunks, model="text-embedding-ada-002"):
    logger.info("Generating embeddings for each chunk.")
    embeddings = []
    for text in text_chunks:
        embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
        embeddings.append(np.array(embedding, dtype=np.float32))
    return embeddings


# Store Embeddings in FAISS
def store_embeddings_in_faiss(embeddings, chunks):
    logger.info("Storing embeddings in FAISS index.")
    dimension = len(embeddings[0])  # Dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance for FAISS index
    metadata = {}

    for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        vector_id = np.array([embedding], dtype=np.float32)
        index.add(vector_id)
        metadata[idx] = chunk  # Store chunk text associated with vector id

    return index, metadata


# Retrieve and Generate Answer using RAG
def retrieve_and_answer(question, index, metadata):
    logger.info("Generating embeddings for the question.")
    question_embedding = client.embeddings.create(input=[question], model="text-embedding-ada-002").data[0].embedding
    question_embedding = np.array([question_embedding], dtype=np.float32)

    logger.info("Retrieving relevant chunks from the FAISS index.")
    k = 3  # Number of similar results to retrieve
    distances, indices = index.search(question_embedding, k)

    # Combine the retrieved chunks into a single context
    relevant_text = " ".join([metadata[idx] for idx in indices[0]])

    logger.info("Generating answer using retrieved chunks.")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"{relevant_text}\nQuestion: {question}"}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content


# Streamlit App
st.title("Document-based Q&A System with RAG")
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
question = st.text_input("Enter your question:")

if uploaded_file and question:
    try:
        logger.info("Processing uploaded file and question.")
        text = parse_pdf(uploaded_file)
        chunks = chunk_text(text)
        embeddings = generate_embeddings(chunks)
        index, metadata = store_embeddings_in_faiss(embeddings, chunks)

        answer = retrieve_and_answer(question, index, metadata)
        st.write("Answer:", answer)
    except Exception as e:
        logger.error("An error occurred during processing", exc_info=True)
        st.error(f"An error occurred: {e}")
