from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
import os
import google.generativeai as genai
import textwrap

from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def process_docx(docx_file):
    # Use Docx2txtLoader to load the Document
    loader = Docx2txtLoader(docx_file)
    # Load Documents and split into chunks
    documents = loader.load_and_split()
    return documents

def process_pdf(pdf_file):
    # Use PyPDFLoader to load PDF documents from a file-like object
    loader = PyPDFLoader(file_path=pdf_file)
    documents = loader.load_and_split()
    return documents

def main():
    st.title("CV Summary Generator")

    # Upload file
    uploaded_file = st.file_uploader("Select CV", type=["docx", "pdf"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]

        st.write("File Details:")
        st.write(f"File Name: {uploaded_file.name}")
        st.write(f"File Type: {file_extension}")

        # Process the file based on its type
        if file_extension == "docx":
            # Pass the file object to process_docx
            documents = process_docx(uploaded_file)
        elif file_extension == "pdf":
            # Pass the file object to process_pdf
            documents = process_pdf(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a .docx or .pdf file.")
            return

        # Initialize Google Gemini LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

        # Define prompt templates for the LLM
        prompt_template = """You have been given a Resume to analyse. 
        Write a verbose detail of the following: 
        {text}
        Details:"""
        prompt = PromptTemplate.from_template(prompt_template)

        refine_template = (
            "Your job is to produce a final outcome\n"
            "We have provided an existing detail: {existing_answer}\n"
            "We want a refined version of the existing detail based on initial details below\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original summary in the following manner:"
            "Name: \n"
            "Email: \n"
            "Key Skills: \n"
            "Last Company: \n"
            "Experience Summary: \n"
        )
        refine_prompt = PromptTemplate.from_template(refine_template)

        # Load summarize chain for the LLM
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )

        # Run the chain to summarize the documents
        result = chain({"input_documents": documents}, return_only_outputs=True)

        # Display the result
        st.write("Resume Summary:")
        st.text_area("Text", result['output_text'], height=400)

if __name__ == "__main__":
    main()
