import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import torch.nn.functional as F
import streamlit as st
import pandas as pd
import io
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Set dark theme globally via config.toml, or simulate it here
st.markdown("""
    <style>
    body, .stApp { background-color: #121212; color: #e0e0e0; }
    textarea, .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: white;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #0a84ff;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
model_path = "./fake-news-bert-20250522T140643Z-1-001/fake-news-bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Fake News Detector")
st.markdown("Enter a news article or headline to determine its authenticity.")

# Sidebar info
with st.sidebar:
    st.header("Model Details")
    st.markdown("""
    **Model:** DistilBERT  
    **Frameworks:** Transformers, PyTorch, Streamlit  
    **Built by:** Ayush Ahirwar
    """)

# User input text area
user_input = st.text_area("Paste your news content here:", height=150)

# Analyze button
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing with BERT..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item() * 100

            label = "Real" if pred == 1 else "Fake"
            color = "green" if pred == 1 else "red"

            st.markdown(f"<h3 style='color:{color};'>Prediction: {label}</h3>", unsafe_allow_html=True)
            st.progress(confidence / 100)
            st.markdown(f"<p style='font-size: 16px;'>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)

            with st.expander("Why this result?"):
                st.write("This prediction is based on patterns learned from thousands of real and fake news articles.")

            # Save to history
            st.session_state.history.append({
                "Input": user_input,
                "Prediction": label,
                "Confidence": f"{confidence:.2f}%"
            })

# Show history and download option
if st.session_state.history:
    st.markdown("---")
    st.header("History")

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    csv = history_df.to_csv(index=False)
    st.download_button(
        label="Download History as CSV",
        data=csv,
        file_name="fake_news_detection_history.csv",
        mime="text/csv"
    )