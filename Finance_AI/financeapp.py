import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from PyPDF2 import PdfReader
import pdfplumber
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Set up the API key for Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Google LLM and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract tables from PDFs
def get_pdf_tables(pdf_docs):
    tables = []
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
    return tables

# Function to chunk the document text
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Function to create a prompt for the LLM
def create_prompt(extracted_text, extracted_tables):
    prompt = f"""
    Based on the following financial data extracted from the PDF, please calculate the key financial metrics and provide a brief analysis:

    1. Profitability (e.g., Net Profit Margin, Return on Equity)
    2. Liquidity (e.g., Current Ratio, Quick Ratio)
    3. Solvency (e.g., Debt to Equity Ratio)
    4. Efficiency (e.g., Asset Turnover Ratio)

    Here is the extracted data from the financial statement:

    Text Data:
    {extracted_text}

    Table Data (if any):
    {extracted_tables if extracted_tables else "No table data extracted"}
    Give detailed report analysis for managment and stake holder review
    Please compile a brief report summarizing the key findings about 1 or 2 pages from the analysis.
    """
    return prompt

# Function to create a vector store from chunks
def create_vector_store(text_chunks):
    # Generate embeddings for each chunk
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

# Streamlit UI
st.title("Financial Statement Analyzer")

uploaded_file = st.file_uploader("Upload a financial statement (PDF only)", type=["pdf"])

if uploaded_file is not None:
    # Extract text and table from PDF
    extracted_text = get_pdf_text([uploaded_file])
    extracted_table = get_pdf_tables([uploaded_file])

    # Chunk the extracted text
    text_chunks = chunk_text(extracted_text)

    # Create a vector store using the chunks
    vector_store = create_vector_store(text_chunks)

    # Retrieve relevant chunks for analysis
    relevant_chunks = vector_store.similarity_search(extracted_text, k=4)
    combined_text = " ".join([chunk.page_content if hasattr(chunk, "page_content") else str(chunk) for chunk in relevant_chunks])

    # Create a prompt based on the combined relevant text and tables
    prompt = create_prompt(combined_text, extracted_table)

    # Call the LLM to analyze the data
    st.subheader("Analyzing...")

    try:
        # Use LLMChain to handle the analysis
        chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
        analysis = chain.run({"cv_text": combined_text})
        st.subheader("Analysis Result:")
        st.write(analysis)

        # Option to download the report
        st.download_button(
            label="Download Report",
            data=analysis,
            file_name="financial_analysis_report.txt",
            mime="text/plain",
        )

    except Exception as e:
        st.error(f"An error occurred while analyzing the document: {str(e)}")
