import streamlit as st
from PIL import Image

from imagecaption import generate_caption
from model_loader import load_bilstm_pipeline, predict_toxicity
from database import log_to_db, get_all_records

st.set_page_config(page_title="Toxic Content Classifier", layout="wide")

# Cache model and vocab in Streamlit memory so it loads instantly
@st.cache_resource
def load_pipeline():
    return load_bilstm_pipeline()

model, vocab = load_pipeline()

st.title("🛡️ Toxic Content Classification System")

tab1, tab2 = st.tabs(["Classify Input", "View Database Records"])

# tab 1: CLASSIFICATION INTERFACE 
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
            with st.spinner("Evaluating toxicity with BiLSTM..."):
                preds = predict_toxicity(input_text, model, vocab)
                log_to_db(input_type, input_text, preds)
                st.success("Analysis complete and result saved to CSV database!")
                
                
                cols = st.columns(6)
                for idx, (label, info) in enumerate(preds.items()):
                    with cols[idx]:
                        status = "🚨 TOXIC" if info['is_toxic'] else "✅ CLEAN"
                        st.metric(
                            label=label.replace('_', ' ').title(), 
                            value=status, 
                            delta=f"{info['probability']*100:.1f}%"
                        )

# tab 2: DATABASE VIEWER 
with tab2:
    st.subheader("Database Audit Trail (`classification_database.csv`)")
    if st.button("Refresh Table"):
        st.rerun()
        
    df = get_all_records()
    if not df.empty:
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No records currently logged in the database.")