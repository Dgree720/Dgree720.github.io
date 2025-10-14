#Filename: 16_3LLM03.py
import datetime
from openai import OpenAI
import os
import openai
from dotenv import load_dotenv
load_dotenv(override=True)
# Retrieve the API keys from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key= openai.api_key )

response = client.images.generate(
    model="dall-e-2",
    prompt="a white siamese cat",
    size="512x512",
    n=1,
)

print(response.data[0].url)

print(" ")
print('16_3LLM03'+ ' | Student ID | ', datetime.datetime.now())