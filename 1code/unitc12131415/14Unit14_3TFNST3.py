# streamlit run 14Unit14_3TFNST3.py
import tensorflow_hub as hub
import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
st.title('Unit14_3 | StudentID | Arbitrary Image Stylization')
source_name = st.sidebar.selectbox('📷 source photo', ('g1.jpg', 'g2.jpg', 'g3.jpg','3.png','bgc.jpg','cat_and_dog.jpg','IMG_2997.jpg','person1.png'))
style_name = st.sidebar.selectbox('🎭 style photo ', ("ss.jpg", "ss2.jpg", "ss3.jpg",'g1.jpg', 'g2.jpg', 'g3.jpg',"andy.jpg", "candy.jpg",
                            "composition.jpg", "la_muse.jpg","mosaic.jpg", "starry_night.jpg", "the_wave.jpg" ))
source_image=cv2.imread("pic/" + source_name)
source_image = cv2.resize(source_image, (480, 280))
style_image=cv2.imread("models/" + style_name)
col1, col2 = st.columns(2)
with col1:
   st.header("🖼️ Source image:")
   st.image(source_image, channels= 'BGR')
with col2:
   st.header('🎨 Style image:')
   st.image(style_image, channels= 'BGR')
clicked = st.button('✨ Style Transfer')
if clicked:
    source_image = source_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, (256, 256))
    model = hub.load('models/')
    out_image = model(tf.constant(source_image), tf.constant(style_image))
    st.write('🎉 Output image:')
    st.image(np.squeeze(out_image), width=400 , channels= 'BGR')