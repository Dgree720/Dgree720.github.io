# streamlit run 16_6LLM06.py
import os
import openai
import streamlit as st
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(override=True)
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai.api_key)

st.title("gpt-5-nano LLM Story Telling")
if st.button("Storytelling"):
    # Create an empty text container
    story_placeholder = st.empty()
    story = ""

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "user", "content": "Please write a bedtime story approximately 100 words in length"},
        ],
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            story += chunk.choices[0].delta.content
            story_placeholder.write(story)
