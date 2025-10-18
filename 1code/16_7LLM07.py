# streamlit run 16_7LLM07.py
import os
import openai
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Retrieve the API keys from the environment variable
load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

st.title("🗣️ LLM Story Telling + GPT-5-nano voice 16_7LLM07"+ "| Student ID |")

if st.button("Storytelling"):
    story_placeholder = st.empty()
    story = ""

    # Generate the story text
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

    # Generating audio (GPT-4o Audio Preview)
    with st.spinner("Generating audio..."):
        speech_response = openai.audio.speech.create(
            model="gpt-4o-mini-tts",  # Speech Synthesis Model
            voice="nova",   # options:   "nova", "shimmer", "echo", "fable", "onyx"
            input=story,
        )
        audio_path = "story.mp3"
        with open(audio_path, "wb") as f:
            f.write(speech_response.content)

        st.audio(audio_path, format="audio/mp3")
