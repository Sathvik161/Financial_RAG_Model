import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import pandas as pd
import pdfplumber
import tempfile
from langchain_groq import ChatGroq
import camelot

# Load API keys from .env file
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"

# Streamlit Title
st.title("Financial Document Q&A with RAG")

# Initialize the language model (ChatGroq in this case)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="Llama3-8b-8192"
)

# Function to extract tables using pdfplumber
def extract_pl_table_from_pdf(pdf_file):
    """
    Extract the specific P&L table from the PDF using Camelot when the table is found,
    ignoring the contents section.
    """
    pl_data = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                
                # Check if the table header text is in the current page, but we don't want the contents
                if "Condensed Consolidated Statement of Profit and Loss" in text:
                    # Use Camelot to extract tables from the current page where the table is located
                    tables = camelot.read_pdf(pdf_file, pages=str(page_num), flavor='stream')

                    # If Camelot finds any tables, use the first one (ignoring the header section)
                    if tables:
                        # Assuming the first table contains the P&L data
                        pl_data = tables[0].df

                        # Convert table into a list of dictionaries (skip headers if necessary)
                        headers = pl_data.iloc[0].tolist()  # First row as headers
                        pl_data = pl_data.iloc[1:].apply(lambda row: dict(zip(headers, row)), axis=1).tolist()

        if not pl_data:
            raise ValueError("P&L table not found in the specified section.")
    except Exception as e:
        raise
    return pd.DataFrame(pl_data)

# Function to embed documents and create a vector store
def vector_embedding(temp_file_path):
    if "vectors" not in st.session_state:
        # Initialize embeddings and loader
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
        st.session_state.loader = PyPDFLoader(temp_file_path)
        st.session_state.docs = st.session_state.loader.load()

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = splitter.split_documents(st.session_state.docs)

        # Generate embeddings and store in FAISS
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )
        st.success("Vector Store DB is ready!")

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""
As a financial analyst, use the provided data to accurately answer the following question. If calculations are needed, perform them and provide a clear, concise response with appropriate references
Reason with data if needed.
Context:
{context}

Question: {input}

Answer:
""")

# File Upload and Temporary File Handling
uploaded_file = st.file_uploader("Upload Financial PDF", type="pdf")
if uploaded_file:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Extract P&L Table
    pl_table_df = extract_pl_table_from_pdf(temp_file_path)
    if pl_table_df is not None:
        st.write("Extracted P&L Table:")
        st.dataframe(pl_table_df)
    else:
        st.warning("No tables were found in the uploaded PDF.")

    # Embed Documents Button
    if st.button("Embed Documents"):
        vector_embedding(temp_file_path)

# Query Input
user_query = st.text_input("Enter your financial query")
if st.button("Submit Query"):
    if "vectors" not in st.session_state:
        st.error("Please embed the documents first!")
    else:
        # Create the RAG pipeline
        llm_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, llm_chain)

        # Get response
        response = retrieval_chain.invoke({"input": user_query})
        st.write("Answer:", response["answer"])

        # Show relevant documents
        with st.expander("Relevant Context"):
            for doc in response["context"]:
                st.write(doc.page_content)
