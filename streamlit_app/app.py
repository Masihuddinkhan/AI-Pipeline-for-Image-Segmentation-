import streamlit as st
import os  
import streamlit as st
from PIL import Image
import pandas as pd

st.title("AI Pipeline: Image Segmentation and Analysis")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image (call segmentation, identification, etc.)
    # Display segmented objects, descriptions, extracted text, and summaries

    if st.button("Show Final Output"):
        output_image_path = r'D:\AI Pipeline for Image Segmentation\data\output\final_image.png'
        print("Trying to open image at:", output_image_path)
        if os.path.exists(output_image_path):
            st.image(output_image_path)
            st.write(pd.read_csv('data/output/summary_table.csv'))  # Summary table
        else:
            st.error("Final image not found. Please ensure that the pipeline ran successfully.")

