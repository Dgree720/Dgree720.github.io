# streamlit run 11Unit11_3llm3.py
import streamlit as st
import ollama
import random

st.title("Gemma3 Multimodal LLM examples Unit11_3 | 322022 ")
colors = ["red", "blue", "green", "orange", "purple"]
user_input = st.text_input("You can ask any question:", "")
uploaded_image = st.file_uploader(
    "Images can also be uploaded（jpg/png）", type=["jpg", "jpeg", "png"]
)

if st.button("send"):
    if user_input or uploaded_image:
        st.markdown(
            f"<span style='color:{random.choice(colors)};'>Please wait a moment, thinking…</span>",
            unsafe_allow_html=True,
        )
        messages = []
        if uploaded_image:
            st.image(
                uploaded_image,
                caption="The uploaded image is shown above. Please wait for the analysis results...",
                width="stretch",
            )
            image_bytes = uploaded_image.getvalue()
            messages.append(
                {"role": "user", "content": user_input, "images": [image_bytes]}
            )
        else:
            messages.append({"role": "user", "content": user_input})
        response = ollama.chat(model="gemma3:4b", messages=messages)
        st.write(response["message"]["content"])
    else:
        st.warning("Please enter a prompt or upload an image!")
