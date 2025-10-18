# Filename: 16_1LLM01.py
import os
import openai
import datetime
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
# Retrieve the API keys from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

completion = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {
            "role": "user",
            "content": "Please write a unicorn bedtime story in a single sentence.",
        }
    ],
)
print(completion)
print("gpt-5-nano bedtime story: " + completion.choices[0].message.content)
# Assignment setting: Please enter your Student ID.
print(" ")
print("16_1LLM01" + " | 322022 | ", datetime.datetime.now())
