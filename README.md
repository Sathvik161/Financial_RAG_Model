# Financial Document Q&A with RAG

Unlock insights from financial documents using an AI-powered Retrieval-Augmented Generation (RAG) pipeline. This application leverages Streamlit for an intuitive user interface, Google Generative AI for embeddings, and ChatGroq for language model-based question answering. It processes financial PDFs, converts them into vector embeddings, and provides context-aware answers to user queries.

## Features
- Upload financial documents (PDFs).
- Extract document content and convert it into embeddings using LlamaParse and Google Generative AI Embeddings.
- Process user queries with ChatGroq and provide data-driven answers.
- Display relevant document context for transparency.

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: LangChain for embedding and retrieval
- **AI Models**: ChatGroq and Google Generative AI
- **Document Parsing**: LlamaParse
- **Vector Store**: FAISS

## Prerequisites
- Python 3.9 or later
- API keys for:
  - Google Generative AI (GOOGLE_API_KEY)
  - ChatGroq (GROQ_API_KEY)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/Financial_RAG_Model.git
   cd Financial_RAG_Model
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload a PDF:**
   - Use the file uploader to upload a financial document in PDF format.
   - The document will be processed and converted into embeddings automatically.

2. **Ask Questions:**
   - Enter a financial query in the text input field.
   - Submit the query to get a context-aware answer from the RAG pipeline.

3. **View Relevant Context:**
   - Expand the "Relevant Context" section to see the parts of the document that were used to answer the query.

## Project Structure
```
Financial_RAG_Model/
├── FinalRAG.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration for deployment
├── .env                # Environment variables
├── rag_model.ipnby      # jupiter notebook 
└── README.md           # Project documentation
```

## Example Query
1. Upload a financial document, e.g., "Company_Annual_Report.pdf".
2. Enter a query such as:
   - "What is the company's net profit for 2023?"
   - "List the key financial ratios for the last quarter."

## Docker Support

To run the application using Docker:

1. **Build the Docker Image:**
   ```bash
   docker build -t financial-rag .
   ```

2. **Run the Docker Container:**
   ```bash
   docker run -p 8501:8501 --env-file .env financial-rag
   ```

3. Open the application in your browser at `http://localhost:8501`.

## Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your fork:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or feedback, please open an issue or contact the project maintainer at <sathvik.vittapu@gmail.com>.

