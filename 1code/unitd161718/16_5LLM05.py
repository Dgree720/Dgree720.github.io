# 16_5LLM05.py
import os
import openai
import datetime
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv(override=True)
# Retrieve the API keys from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {
            "role": "user",
            "content": "Tell me a science fiction story approximately 100 words in length.",
        },
    ],
    stream=True,
)

for chunk in response:
    print(chunk)
    print(chunk.choices[0].delta)
    print("****************")

# Assignment setting: Please enter your Student ID.
print(" ")
print("16_4LLM04" + " | 322022 | ", datetime.datetime.now())
