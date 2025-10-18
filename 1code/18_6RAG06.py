# streamlit run 18_6RAG06.py
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

llm = OllamaLLM(model='gemma3:1b', temperature=0.7)

st.title('Toolbox 18_6RAG06' + '|Student ID|')
role = st.selectbox("What role would you like the AI to play?",
                    ('Senior Python Developer', 'Professional English Teacher'),
                    index=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Your job is to answer user questions in English."),
    ("user", "{input}"),
])

chain = prompt | llm

def generate_response(text):
    for r in chain.stream({'input': text, 'role': role}):
        yield r

template = 'None'
if role == 'Senior Python Developer':
    template = st.radio("Template", ['Naming', 'Code Review', 'None'], horizontal=True)
elif role == 'Professional English Teacher':
    template = st.radio("Template",
                       ['Grammar Correction (with explanation)', 'Grammar Correction', 'None'],
                       horizontal=True)

def use_template(text):
    if template == 'None':
        return text
    if template == 'Naming':
        return f'Is "{text}" a good Python variable name?'
    if template == 'Code Review':
        return (
            "Please help me review the following code snippet. "
            "Please point out any issues and provide additional explanations. "
            "The code snippet is as follows:\n"
            f"'''\n{text}\n'''"
        )
    if template == 'Grammar Correction (with explanation)':
        return (
            "Please help me correct the grammatical errors in the following text and provide explanations. "
            "The text is as follows:\n"
            f"'''\n{text}\n'''"
        )
    if template == 'Grammar Correction':
        return (
            "Please help me correct the grammatical errors in the following text. "
            "You do not need to provide additional explanations. "
            "The text is as follows:\n"
            f"'''\n{text}\n'''"
        )

# Create input form and display response
with st.form('form'):
    text = st.text_area('Enter text:', '')
    submitted = st.form_submit_button('Send')
    if submitted:
        st.write_stream(generate_response(use_template(text)))