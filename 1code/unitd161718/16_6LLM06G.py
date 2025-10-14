# streamlit run 16_6LLM06G.py
import os
import streamlit as st
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(override=True)

GEMINI_api_key = os.getenv('GEMINI_API_KEY')
client = OpenAI(
    api_key=GEMINI_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

st.title("gemini-2.5-flash LLM Story Telling")
if st.button("Storytelling"):
    # Create an empty text container
    story_placeholder = st.empty()
    story = ""

    response= client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please write a bedtime story approximately 100 words in length"},
        ],
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            story += chunk.choices[0].delta.content
            story_placeholder.write(story)
