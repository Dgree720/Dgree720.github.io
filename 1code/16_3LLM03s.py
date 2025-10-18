# pip install streamlit
# streamlit run 16_3LLM03s.py
import os
import openai
import streamlit as st

from dotenv import load_dotenv

load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")
# Retrieve the API keys from the environment variable
# Streamlit interface
st.title("🎨 AI Image Generator（DALL·E）16_3LLM03"+ " | Student ID | ")
st.caption("Enter a text prompt, and let OpenAI generate the image for you!")

# prompt
prompt = st.text_input("Please enter the description of what you want to generate:", value="Robot holding a red skateboard")

if st.button("Generate Image"):
    with st.spinner("Image generation in progress; please wait..."):
        try:
            response = openai.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size="512x512",
                n=1,
            )
            image_url = response.data[0].url
            st.image(image_url, caption="🖼️ generated image", width='stretch')
        except Exception as e:
            st.error(f"⚠️ error：{e}")
