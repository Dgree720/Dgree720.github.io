# streamlit run 18_7RAG07.py
# pip install fastembed chromadb
import streamlit as st
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

# --- Prompt Template ---
promptTemplate = """You are an expert assistant. Answer the question as precisely as possible using the provided context. If the provided context does not contain sufficient information, please answer using your internal knowledge.

Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=promptTemplate, input_variables=["context", "question"])

# --- Global Variables ---
vector_store = None
retriever = None
llm = None


# --- Load PDF & Build VectorDB ---
def load_pdf_and_build_db(uploaded_pdf):
    # Save uploaded PDF to temporary folder
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_pdf.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()
    chunks = filter_complex_metadata(pages)

    global vector_store, retriever
    # Create vector database, specify persist_directory for proper table initialization
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=FastEmbedEmbeddings(),
        persist_directory="chroma_db"
    )
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.2}
    )
    return "PDF loaded and vector DB created!"


# --- Chat Function ---
def model_response(message):
    global retriever, llm
    if retriever is None:
        return "Please upload a PDF first!"

    # Use invoke()
    docs = retriever.invoke(message)
    context = "\n".join([doc.page_content for doc in docs]) if docs else ""

    # Integrate retrieval results into prompt
    chain_input = prompt.format(context=context, question=message)

    # Call llm using invoke()
    response = llm.invoke(chain_input)
    return response


def main():
    global llm
    st.title("📚 Case PDF Reading Assistant 18_7RAG07" + '|Student ID|')

    # Initialize LLM (avoid repeated initialization)
    if llm is None:
        llm = OllamaLLM(model='gemma3:1b', temperature=0.7)

    # Use session_state to store conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # PDF file upload
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf is not None:
        status = load_pdf_and_build_db(uploaded_pdf)
        st.success(status)

    # User input question
    user_input = st.text_input("Your question:")
    if st.button("Send") and user_input:
        response = model_response(user_input)
        st.session_state.conversation.append(("User", user_input))
        st.session_state.conversation.append(("Assistant", response))

    # Display conversation
    st.markdown("### Conversation")
    for sender, message in st.session_state.conversation:
        if sender == "User":
            st.markdown(f"**User:** {message}")
        else:
            st.markdown(f"**Assistant:** {message}")


if __name__ == "__main__":
    main()