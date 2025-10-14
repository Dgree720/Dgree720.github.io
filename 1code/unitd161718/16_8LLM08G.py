# streamlit run 16_8LLM08G.py
import os
import base64
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

# Retrieve the API keys from the environment variable
GEMINI_api_key = os.getenv('GEMINI_API_KEY')
client = OpenAI(
    api_key=GEMINI_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
st.title("🖼️ GEMINI_API Image content recognition")
# 上傳圖片
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="image uploaded",  width='stretch')
    # Convert the image to base64
    image_bytes = uploaded_file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Image content recognition using gemini-2.5
    if st.button("Image recognition"):
        with st.spinner("Recognition in progress; please wait..."):
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this image?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
        #  Analysis Results
        st.subheader("🧠 Model Analysis Results:")
        st.write(response.choices[0].message.content)
