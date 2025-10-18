# Filename: 16_0LLM00G.py
import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
# Retrieve the API keys from the environment variable
GEMINI_api_key = os.getenv("GEMINI_API_KEY")
client = OpenAI(
    api_key=GEMINI_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

models = client.models.list()
for model in models:
    print(model.id)
