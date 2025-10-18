# Filename: 16_1LLM01G.py
import os
import datetime
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv(override=True)
# Retrieve the API keys from the environment variable
GEMINI_api_key = os.getenv("GEMINI_API_KEY")
print(GEMINI_api_key)
client = OpenAI(
    api_key=GEMINI_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

completion = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        dict(
            role="user",
            content="Please write a unicorn bedtime story in a single sentence.",
        )
    ],
)
print("Gemini-2.5-flash bedtime story: " + completion.choices[0].message.content)

print(" ")
print("16_1LLM01G" + " | 322022 | ", datetime.datetime.now())
