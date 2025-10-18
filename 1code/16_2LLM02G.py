# pip install requests
# Filename: 16_2LLM02G.py
import os
import base64
import requests
import datetime
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv(override=True)
# Retrieve the API keys from the environment variable
GEMINI_api_key = os.getenv("GEMINI_API_KEY")
client = OpenAI(
    api_key=GEMINI_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
# Download the image and convert it to base64
image_url = "https://www.hs-pforzheim.de/fileadmin/_processed_/0/e/csm_HSPF_Website_0039_4041d8a00e.webp"
image_data = requests.get(image_url).content
image_dl = base64.b64encode(image_data).decode("utf-8")

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe the content of the image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_dl}"},
                },
            ],
        }
    ],
)
print("gemini-2.5-flash: " + response.choices[0].message.content)
print("16_2LLM02G" + " | 322022 | ", datetime.datetime.now())
