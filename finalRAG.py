import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from llama_parse import LlamaParse
from dotenv import load_dotenv
import tempfile

# Load API keys from .env file
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"

# Streamlit Title and Branding
st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            padding: 20px;
        }
        .header {
            color: #2c3e50;
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            padding-bottom: 10px;
        }
        .sub-header {
            text-align: center;
            font-size: 18px;
            margin-bottom: 40px;
            color: #7f8c8d;
        }
        .cta-btn {
            background-color: #3498db;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 12px 20px;
            cursor: pointer;
        }
        .cta-btn:hover {
            background-color: #2980b9;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Financial Document Q&A with RAG")
st.markdown("<p class='sub-header'>Unlock insights from financial documents using AI-powered Q&A.</p>", unsafe_allow_html=True)


# Initialize the language model (ChatGroq in this case)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="Llama3-8b-8192"
)

# Function to process PDF with LlamaParse
def process_pdf_with_llamaparse(pdf_file):
    """
    Use LlamaParse to extract the content of a PDF as markdown.
    Returns a single concatenated markdown string.
    """
    try:
        # LlamaParse returns a list of Document objects
        markdown_data = LlamaParse(result_type="markdown").load_data(pdf_file)
        
        # Extract the 'text' attribute from each Document and join them
        markdown_text = "\n".join(doc.text for doc in markdown_data)

        return markdown_text
    except Exception as e:
        st.error(f"Error extracting markdown with LlamaParse: {e}")
        return None

# Function to embed Markdown text and create a vector store
def vector_embedding_from_markdown(markdown_text):
    if "vectors" not in st.session_state:
        # Initialize embeddings
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
        
        # Split markdown text into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = splitter.split_text(markdown_text)

        # Generate embeddings and store in FAISS
        st.session_state.vectors = FAISS.from_texts(documents, st.session_state.embeddings)
        st.session_state.processing_done = True  # Set processing as complete
        st.success("Processing complete! You can now ask your queries.")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(""" 
You are a financial analyst tasked with answering questions based on the provided financial data. Your answers should be clear, concise, and well-reasoned, using the data to support your conclusions. If any calculations are required, please perform them step-by-step, showing your process. If necessary, break down complex questions into simpler components and reason with the data accordingly.

Please ensure your answer:
1. Refers explicitly to relevant data points or sections from the provided context.
2. Includes any necessary calculations with clear steps and reasoning behind them.
3. Uses data-driven logic to answer, especially when multiple data points are involved.
4. If you need to make any assumptions to answer the question, clearly state them.

Context:
{context}

Question: {input}

Answer:

""")

# File Upload and Temporary File Handling
uploaded_file = st.file_uploader("Upload Financial PDF", type="pdf")
if uploaded_file:
    # Check if the document has already been processed and markdown is available
    if "markdown_text" not in st.session_state:
        # Save uploaded file to a temporary file with .pdf extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Convert PDF to Markdown using LlamaParse
        markdown_text = process_pdf_with_llamaparse(temp_file_path)
        if markdown_text:
            # Store markdown in session state to avoid reprocessing
            st.session_state.markdown_text = markdown_text
            st.success("Document is being processed...")

            # Automatically embed Markdown into Vector Store
            vector_embedding_from_markdown(markdown_text)

# Query Input
user_query = st.text_input("Enter your financial query")
if st.button("Submit Query"):
    if "vectors" not in st.session_state:
        st.error("Please upload and process a document first!")
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
