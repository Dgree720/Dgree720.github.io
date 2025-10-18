# streamlit run 18_4RAG04.py
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="gemma3:1b", temperature=0.7)
# build prompt
prompt = PromptTemplate.from_template("Question: {question}")
# use LCEL（LangChain Expression Language）
chain = prompt | llm | StrOutputParser()
st.title("🧠 Ask Me Anything (Ollama - LangChain) 18_4RAG04" + "| 322022 |")
with st.form("my_form"):
    user_input = st.text_area("Ask Anything：", "")
    submitted = st.form_submit_button("send")
    if submitted and user_input:
        #  execute chain
        with st.spinner("Generating..."):
            st.write_stream(chain.stream({"question": user_input}))
