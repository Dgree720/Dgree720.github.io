#Filename: 16_2LLM02.py
import os
import openai
import datetime
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(override=True)

# Retrieve the API keys from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key= openai.api_key )

response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe the content of the image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.hs-pforzheim.de/fileadmin/_processed_/0/e/csm_HSPF_Website_0039_4041d8a00e.webp",
                    },
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
# Assignment setting: Please enter your Student ID.
print(" ")
print('16_2LLM02'+ ' | Student ID | ', datetime.datetime.now())