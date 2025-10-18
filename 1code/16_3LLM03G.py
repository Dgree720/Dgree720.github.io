#Filename:  16_3LLM03G.py
import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)
GEMINI_api_key = os.getenv('GEMINI_API_KEY')
client = OpenAI(
    api_key=GEMINI_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# Imagen API is only accessible to billed users at this time.
# response = client.images.generate(
#     model="imagen-3.0-generate-002",
#     prompt="a white siamese cat",
#     size="1024x1024",
#     quality="standard",
#     response_format='b64_json',
#     n=1,
# )

# print(response.data[0].url)

# Assignment setting: Please enter your Student ID.
print(" ")
print('W8 Assignment 3'+ ' | Student ID | ', datetime.datetime.now())