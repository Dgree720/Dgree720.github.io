# streamlit run 18_8RAG08.py
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

# Create a directory for ChromaDB persistence
CHROMA_DB_DIR = os.path.join(os.getcwd(), "chroma_db")
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Page configuration
st.set_page_config(page_title="PDF Case Assistant", layout="wide")

# --- Prompt Template ---
promptTemplate = """You are an expert assistant. Answer the question as precisely as possible using the provided context. If not available, say "answer not available in context."

Context: {context}
Question: {question}
Answer:
"""

prompt = PromptTemplate(
    template=promptTemplate, input_variables=["context", "question"]
)

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = OllamaLLM(model="gemma3:1b", temperature=0.7)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# --- Load PDF & Build VectorDB ---
def load_pdf_and_build_db(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        chunks = filter_complex_metadata(pages)

        # Create a unique collection name for this session
        collection_name = f"pdf_collection_{hash(uploaded_pdf.name)}"

        # Initialize ChromaDB with a persistent directory
        st.session_state.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory=CHROMA_DB_DIR,
            collection_name=collection_name,
        )

        st.session_state.retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.2},
        )

        # Mark as processed
        st.session_state.pdf_processed = True

        # Clean up the temporary file
        os.unlink(tmp_file_path)
        return True
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return False


# --- Chat Function ---
def get_model_response(question):
    if not st.session_state.pdf_processed or st.session_state.retriever is None:
        return "Please upload and process a PDF first!"

    try:
        caseChain = (
            {"context": st.session_state.retriever, "question": RunnablePassthrough()}
            | prompt
            | st.session_state.llm
            | StrOutputParser()
        )

        with st.spinner("Thinking..."):
            response = caseChain.invoke(question)

        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"


# --- Main UI ---
st.title("📚 Case PDF Reading Assistant II 18_8RAG08" + "|Student ID|")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                success = load_pdf_and_build_db(uploaded_file)
                if success:
                    st.success("PDF loaded and vector DB created!")
                    # Clear chat history when new document is loaded
                    st.session_state.chat_history = []

# Main chat interface
st.header("Chat with your PDF")

# Display status
if not st.session_state.pdf_processed:
    st.info("Please upload and process a PDF document to start chatting.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_question = st.chat_input("Ask a question about your document...")
if user_question:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # Display user message
    with st.chat_message("user"):
        st.write(user_question)

    # Get and display assistant response
    response = get_model_response(user_question)

    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response)

# Footer
st.markdown("---")
st.caption("Powered by LangChain and Ollama")
