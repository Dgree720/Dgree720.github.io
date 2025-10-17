# pip install ollama
# streamlit run 11Unit11_1llm1.py
import streamlit as st
import ollama
import random

st.title("Gemma3 text LLM app examples Unit11_1 | 322022")
colors = ["red", "blue", "green", "orange", "purple"]
user_input = st.text_input("You can ask any question:", "")

if st.button("send") or user_input:
    st.markdown(
        f"<span style='color:{random.choice(colors)};'>Please wait a moment, thinking… I’ll respond shortly...</span>",
        unsafe_allow_html=True,
    )
    response = ollama.chat(
        model="gemma3:1b", messages=[{"role": "user", "content": user_input}]
    )
    # response = ollama.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': user_input}])
    st.write(response["message"]["content"])
else:
    st.warning("Please type in your question!")
