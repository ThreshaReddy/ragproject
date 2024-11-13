# RAG: Document-based Q&A System with Retrieval-Augmented Generation

RAG is a document-based Q&A system that leverages Azure OpenAI for generating answers to user queries by extracting and analyzing relevant content from uploaded documents. The application supports PDF files and uses FAISS (Facebook AI Similarity Search) for efficient embedding storage and retrieval.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

RAG provides users with an interactive web application to ask questions based on content within uploaded PDF documents. The application uses OpenAI’s `gpt-4o` model to generate responses based on extracted text from document chunks, indexed and retrieved with FAISS for optimal speed and accuracy.

## Features

- Upload PDF documents and process them for text-based content extraction.
- Automatic chunking of document text for efficient embedding generation and retrieval.
- Embedding storage and retrieval using FAISS for efficient similarity searches.
- Answer generation with Azure OpenAI’s `gpt-4o` model based on retrieved, relevant document chunks.

## Requirements

The following libraries and services are required to run this application:

- **Python**: 3.7 or later
- **Libraries**:
  - `streamlit`
  - `openai`
  - `PyPDF2`
  - `faiss-cpu`
  - `python-dotenv`
- **Azure OpenAI**: An account with an API key and endpoint set up for `gpt-4o` and `text-embedding-ada-002` models

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ThreshaReddy/ragproject.git
   cd ragproject
   ```

2. **Set up a Virtual Environment** (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
   ```

3. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. **Azure OpenAI Setup**:
   - Obtain an **API Key** and **Endpoint** for Azure OpenAI.
   - Ensure the deployment of both `text-embedding-ada-002` (for embeddings) and `gpt-4o` (for generating answers) models on your Azure account.

2. **Environment Variables**:
   - Create a `.env` file in the root directory with the following environment variables:
     ```plaintext
     AZURE_OPENAI_API_KEY=your_openai_api_key
     AZURE_OPENAI_ENDPOINT=https://your_openai_endpoint
     ```
   - Replace `your_openai_api_key` with your Azure OpenAI API key and `your_openai_endpoint` with the endpoint URL.

3. **FAISS**:
   - FAISS is automatically configured as an in-memory index in this application setup and requires no additional setup.

## Running the Application

To start the Streamlit application:

```bash
streamlit run app.py
```

This command will launch the application in your default web browser, where you can interact with the user interface to upload a document and ask questions.

## Usage

1. **Upload a PDF Document**: Use the file uploader in the app interface to upload a PDF document.
   
2. **Enter a Question**: After the document is processed, type a question in the input box.

3. **View the Answer**: The app will use FAISS to retrieve relevant chunks from the document and Azure OpenAI’s `gpt-4o` model to generate an answer based on these chunks.

4. **Logs**: All major steps in processing and retrieval are logged in the terminal for tracking the application’s progress and debugging purposes.

## Project Structure

```plaintext
RAG/
│
├── app.py                   # Main application file for Streamlit
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables (ignored in .gitignore)
```

### Explanation of Key Files

- **app.py**: Contains the application code for uploading PDFs, processing text, chunking, generating embeddings, and generating answers through Streamlit.
- **requirements.txt**: Lists all necessary Python libraries.
- **.env**: Stores your Azure OpenAI credentials. **Ensure that this file is not shared or pushed to public repositories.**

## Troubleshooting

- **Issue**: `Could not connect to tenant default_tenant in ChromaDB`
  - **Solution**: This application now uses FAISS instead of ChromaDB. Ensure FAISS is installed with `pip install faiss-cpu`.
  
- **Issue**: `API Key error for Azure OpenAI`
  - **Solution**: Ensure your `.env` file contains the correct API key and endpoint. Verify your deployment names in Azure OpenAI.

- **Streamlit Crashes or Unexpected Behavior**: 
  - Restart the app with `streamlit run app.py`.
  - Review logs in the terminal for detailed error information.

