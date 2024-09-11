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
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
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
    You are a finance expert based on the uploaded financial statements in PDF format, perform a detailed financial analysis to assess the business's overall health, performance, and potential opportunities or risks. Follow these steps:

    1. Extraction and Identification:
       Extract all relevant financial data from the uploaded PDF, including:
       - Revenue
       - Gross Profit
       - Operating Expenses
       - Net Income
       - Earnings Before Interest and Taxes (EBIT)
       - Total Assets
       - Total Liabilities
       - Shareholders' Equity
       - Cash Flow from Operating Activities
       - Cash Flow from Investing Activities
       - Cash Flow from Financing Activities

       Identify trends, anomalies, and patterns within the extracted data.

    2. Metric Analysis with Numbers:
       Perform a detailed analysis of key financial metrics:
       - Profitability Ratios: Analyze ratios like Gross Margin, Net Margin, Return on Equity, and Return on Assets to evaluate the company’s ability to generate profit relative to its revenue and resources.
       - Liquidity Ratios: Examine the Current Ratio, Quick Ratio, and Cash Ratio to determine the company's ability to cover its short-term liabilities with its short-term assets.
       - Leverage Ratios: Evaluate Debt-to-Equity Ratio, Debt-to-Assets Ratio, and Interest Coverage Ratio to assess the company’s financial leverage and risk exposure.
       - Efficiency Ratios: Calculate Inventory Turnover, Accounts Receivable Turnover, and Asset Turnover to measure how efficiently the company is utilizing its assets to generate revenue.
       - Cash Flow Analysis: Analyze cash flows from operating, investing, and financing activities to determine the cash health of the company.
         
       Analyze and display each key financial metric, showing both the actual numbers and the analysis. Include calculations for profitability, liquidity, leverage, and efficiency ratios. Identify trends, anomalies, and assess financial health based on these numbers.

    3. Report Generation:
       Create a brief report that includes:
       - The actual numbers for each metric.
       - A summary of key findings, insights, recommendations, and potential risks/opportunities.
       - Ensure the report is ready for presentation to management and stakeholders.

       Ensure the report is concise, clear, and suitable for presentation to management and stakeholders.

    4. Output:
       Generate a professional, automatically formatted PDF report that combines both detailed analysis and an executive summary. The report should be suitable for corporate presentations and include visual aids for a more engaging presentation.

    Constraints:
    - The report must be formatted with a professional design and be ready for immediate use by management and stakeholders.
    - Ensure that the analysis is accurate, data-driven, and presented in a way that is accessible to both financial and non-financial stakeholders.

    Instructions:
     Use the extracted data and guidelines above to complete the analysis and generate the final report.
    """
    return prompt

# Function to create a vector store from chunks
def create_vector_store(text_chunks):
    # Generate embeddings for each chunk
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

# Streamlit UI
st.title("Financial Statement Analyzer")

uploaded_files = st.file_uploader("Upload financial statement(s) (PDF only)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Submit & Process"):
        # Extract text and table from PDF
        extracted_texts = [get_pdf_text([file]) for file in uploaded_files]
        extracted_tables = [get_pdf_tables([file]) for file in uploaded_files]

        # Combine extracted texts for chunking
        combined_text = " ".join(extracted_texts)
        
        # Chunk the combined extracted text
        text_chunks = chunk_text(combined_text)

        # Create a vector store using the chunks
        vector_store = create_vector_store(text_chunks)

        # Retrieve relevant chunks for analysis
        relevant_chunks = vector_store.similarity_search(combined_text, k=4)
        combined_relevant_text = " ".join([chunk.page_content if hasattr(chunk, "page_content") else str(chunk) for chunk in relevant_chunks])

        # Create a prompt based on the combined relevant text and tables
        prompt = create_prompt(combined_relevant_text, extracted_tables)

        # Call the LLM to analyze the data
        st.subheader("Analyzing...")

        try:
            # Use LLMChain to handle the analysis
            chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
            analysis = chain.run({"cv_text": combined_relevant_text})
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
