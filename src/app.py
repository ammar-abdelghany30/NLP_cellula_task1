import sys
import os
import streamlit as st
from PIL import Image

from model_loader import load_bilstm_pipeline, predict_toxicity, Vocabulary
from imagecaption import generate_caption
from database import log_to_db, get_all_records

# Fix for Jupyter Pickle Unpickling in Streamlit
import __main__
__main__.Vocabulary = Vocabulary

st.set_page_config(page_title="Toxic Content Classifier", layout="wide")

@st.cache_resource
def load_pipeline():
    return load_bilstm_pipeline()

model, vocab = load_pipeline()

st.title("🛡️️ Multimodal Toxic Content Classification System")

tab1, tab2 = st.tabs(["Classify Input", "View Database Records"])

# --- TAB 1: CLASSIFICATION INTERFACE ---
with tab1:
    st.subheader("Input Text or Upload Image")
    input_type = st.radio("Select Input Format:", ["Direct Text", "Image File"])
    
    input_text = ""
    
    if input_type == "Direct Text":
        input_text = st.text_area("Enter comment to evaluate:", height=100)
    else:
        uploaded_file = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image Preview", width=300)
            
            with st.spinner("Generating Image Caption with BLIP..."):
                input_text = generate_caption(uploaded_file)
                st.info(f"**Generated Caption:** {input_text}")

    if st.button("Classify Content"):
        if not input_text.strip():
            st.warning("Please provide valid text or an image input.")
        else:
            with st.spinner("Evaluating toxicity..."):
                pred_result = predict_toxicity(input_text, model, vocab)
                log_to_db(input_type, input_text, pred_result)
                
                st.write("---")
                st.subheader("Classification Outcome")
                
                if pred_result['status'] == 'TOXIC':
                    st.error(f"🚨 **Overall Output:** {pred_result['summary']}")
                    st.write(f"**Detected Toxic Categories:** {', '.join(pred_result['triggered_labels'])}")
                else:
                    st.success("✅ **Overall Output:** CLEAN")
                    
                st.caption("Result successfully logged to CSV database!")

# --- TAB 2: DATABASE VIEWER ---
with tab2:
    st.subheader("Database Audit Trail (`classification_database.csv`)")
    if st.button("Refresh Table"):
        st.rerun()
        
    df = get_all_records()
    if not df.empty:
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No records currently logged in the database.")