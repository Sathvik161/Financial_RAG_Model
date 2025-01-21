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
        print(f"Error extracting markdown with LlamaParse: {e}")
        return None

# Function to embed Markdown text and create a vector store
def vector_embedding_from_markdown(markdown_text):
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
    
    # Split markdown text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_text(markdown_text)

    # Generate embeddings and store in FAISS
    vectors = FAISS.from_texts(documents, embeddings)
    
    # Return the vectors for use in retrieval
    return vectors

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

# Function to process the uploaded PDF, extract text, and return the answer to the query
def process_pdf_and_query(pdf_file_path, user_query):
    # Convert PDF to Markdown using LlamaParse
    markdown_text = process_pdf_with_llamaparse(pdf_file_path)
    if markdown_text:
        # Embed the Markdown into the vector store
        vectors = vector_embedding_from_markdown(markdown_text)

        # Create the RAG pipeline
        llm_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, llm_chain)

        # Get the response to the user query
        response = retrieval_chain.invoke({"input": user_query})

        # Return the answer
        return response["answer"]
    else:
        return "Error processing PDF."

# Example usage:
pdf_file_path = "Sample Financial Statement.pdf"  # Provide the path to your PDF
user_query = "What are the total expenses for Q2 2023?"

# Get the answer for the user's query
answer = process_pdf_and_query(pdf_file_path, user_query)

# Print the answer
print("Answer:", answer)
