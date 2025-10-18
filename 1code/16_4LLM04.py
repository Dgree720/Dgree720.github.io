# 16_4LLM04.py
import os
import openai
import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
# Retrieve the API keys from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)
story = ""
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {
            "role": "user",
            "content": "Please write a bedtime story, suitable for children, with a length of approximately 300 characters ",
        },
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        story += chunk.choices[0].delta.content
print(story)

speech_response = openai.audio.speech.create(
    model="gpt-4o-mini-tts",  # Speech Synthesis Model: gpt-4o-mini-tts
    voice="nova",  # options:  "nova", "shimmer", "echo", "fable", "onyx"
    input=story,
)
if not os.path.exists("audio"):
    os.makedirs("audio")
with open("audio/story.mp3", "wb") as f:
    f.write(speech_response.content)
# Assignment setting: Please enter your Student ID.
print(" ")
print("16_4LLM04" + " | 322022 | ", datetime.datetime.now())
