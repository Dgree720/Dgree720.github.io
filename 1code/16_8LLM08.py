# streamlit run 16_8LLM08.py
import os
import openai
import streamlit as st
from dotenv import load_dotenv
import base64

# Retrieve the API keys from the environment variable
load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit title
st.title("🖼️ Image content recognition（gpt-5-nano）")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="image uploaded ", width="stretch")

    # Convert the image to base64
    image_bytes = uploaded_file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Image content recognition using the GPT-5-nano
    if st.button("Image recognition "):
        with st.spinner("Recognition in progress; please wait..."):
            response = openai.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
            )
        #  Analysis Results
        st.subheader("🧠 Model Analysis Results:")
        st.write(response.choices[0].message.content)
