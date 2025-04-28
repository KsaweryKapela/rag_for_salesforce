# RAG for Salesforce
### Overview

This app provides a simple Retrieval-Augmented Generation (RAG) system using Streamlit, ChromaDB, and Gemini.
It enables conversational querying over a set of Salesforce earnings call PDFs.
### App flow

- Loads PDFs and chunks them into ~300 token segments (full sentences, sliding window).

- Stores chunks in a local Chroma vector database.

- Streamlit frontend for conversational interaction.

- Gemini LLM decides if query requires searching documents or just returns simple answer.

- If needed, the app retrieves 20 relevant passages and feeds them back into the LLM for a final answer.

- Full chat history is maintained across up to 5 interactions.

### Requirements

[PDF files names from here](https://altimetrik-recruiting-technical-assessment-assets.s3.us-east-1.amazonaws.com/Earnings%20Call%20Transcripts.zip)
must follow format below, change names for it to work properly:

    Salesforce-Inc-Q1-2025-Earnings-Call-May-29-2024.pdf

    (Filename formatting can be automated with an LLM in the future.)

Set up the environment variables from env.example file or at least:

        GEMINI_API_KEY

        DOCS_PATH (path to your PDFs)

### Setup Instructions

Create and activate a virtual environment from main dir:

    python3 -m venv venv
    source venv/bin/activate   # On Linux/Mac
    venv\Scripts\activate      # On Windows

Install the required packages:

    pip install -r requirements.txt

Run the app:

        streamlit run app.py

For auto-reloading on save (debug mode):

        streamlit run app.py --server.runOnSave true

### Tech Stack

Streamlit – Frontend UI

ChromaDB – Vector database

Gemini LLM – Language model for reasoning and answering