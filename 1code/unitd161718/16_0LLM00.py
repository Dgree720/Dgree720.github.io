# Filename: 16_0LLM00.py
# pip install python-dotenv openai

from dotenv import load_dotenv

load_dotenv()

import os
import openai
# Retrieve the API keys from the environment variable

openai.api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY:" + openai.api_key)

GEMINI_api_key = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY:" + GEMINI_api_key)
