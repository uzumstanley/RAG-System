# RAG System Powered by DeepSeek-R1 LLM Model

## Overview
This project is a **Retrieval-Augmented Generation (RAG) System** that allows users to upload PDF files, process their contents, and interact with them through conversational queries. The app leverages **Groq's Mixtral-8x7b LLM model** (DeepSeek-R1 variant) for advanced natural language understanding and answering questions with high accuracy. The backend features FAISS for vector search and OpenAI embeddings for efficient similarity-based retrieval.

---

## Key Features
1. **PDF Upload and Processing**: Upload multiple PDF files, extract their text, and store it in a searchable FAISS vector database.
2. **Conversational Querying**: Ask natural language questions about the content of the uploaded PDFs.
3. **Accurate Answers**: The app uses the DeepSeek-R1 model from Groq, which provides detailed answers, or informs when the information is not available in the context.
4. **RAG Workflow**: Combines information retrieval with generative capabilities for seamless and reliable responses.
5. **Streamlit Interface**: A user-friendly interface for uploading files, processing text, and interacting with the system.

---

## Technology Stack

### Backend
- **LangChain**: Framework for building LLM-powered workflows.
- **FAISS**: Scalable vector database for similarity search.
- **OpenAI Embeddings**: Used for creating embeddings from the extracted text.
- **Groq LLM**: High-performing language model for answering questions.

### Frontend
- **Streamlit**: Simplified app creation for interactive dashboards and interfaces.

### Deployment
- Environment management via `dotenv`.

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.9+
- Pip
- A valid Groq API key

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your Groq API key:
     ```env
     GROQ_API_KEY=your-groq-api-key
     ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage

### Uploading and Processing PDFs
1. Use the **sidebar** to upload one or more PDF files.
2. Click on **Submit & Process** to extract text and create a searchable vector database.
3. Wait for the success message indicating the PDFs have been indexed.

### Asking Questions
1. Enter your question in the text input field on the main page.
2. The app will retrieve relevant information from the vector database and generate an accurate response.

---

## File Structure
```
.
├── app.py                     # Main Streamlit application
├── requirements.txt           # Required Python libraries
├── .env                       # Environment variables (Groq API key)
├── faiss_index/               # Folder containing the FAISS vector store
└── README.md                  # Project documentation
```

---

## Example Scenarios

### Use Case 1: Research Assistance
Upload research papers in PDF format and ask detailed questions about their content.

**Example Question:** "What are the main findings of the study on page 3?"

### Use Case 2: Legal Document Parsing
Upload contracts or legal documents to extract critical clauses or terms.

**Example Question:** "What is the termination clause in this agreement?"

### Use Case 3: Educational Support
Upload textbooks or study material to quickly find answers to specific topics.

**Example Question:** "Explain the concept of retrieval-augmented generation."

---

## Future Enhancements
1. **Support for Additional File Types**: Include support for Word, Excel, or plain text files.
2. **Customizable Chunk Size**: Allow users to customize chunking parameters for optimized retrieval.
3. **Cloud Integration**: Enable cloud storage and retrieval for larger datasets.
4. **Multimodal Querying**: Add support for images and diagrams in PDFs.

---

## Acknowledgments
- **Groq**: For their Mixtral-8x7b model, which powers the app's conversational capabilities.
- **LangChain**: For providing a robust framework for integrating LLMs and vector databases.
- **FAISS**: For efficient and scalable similarity search.

---

## License
This project is licensed under the MIT License.

---

## Contact
For questions or feedback, please reach out to:
- **Name**: Siddharth Kharche
- **Email**: siddukharche04@gmail.com
- **GitHub**: [siddharth-Kharche](https://github.com/siddharth-Kharche)

---

**Happy Conversing!** :books: :speech_balloon:
