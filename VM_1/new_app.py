# app.py  (replace your streamlit app file with this)
import streamlit as st
import pandas as pd
import json
import io
from new_claim import ClaimsNormalizer  # use local module
import matplotlib.pyplot as plt

st.set_page_config(page_title="Claims Description Normalizer", layout="wide", page_icon="📋")

# Simple CSS
st.markdown("""
    <style>
    .main-header { font-size:32px; font-weight:700; }
    .sub-header { color: #6c757d; margin-top: -8px; }
    .result-card { background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_normalizer(use_spacy=True, use_ml=False):
    return ClaimsNormalizer(use_spacy=use_spacy, use_ml=use_ml)

# st.sidebar.image("assets/logo.jpg" if st._is_running_with_streamlit else None, width=150)

import os

if os.path.exists("assets/logo.jpg"):
    st.sidebar.image("assets/logo.jpg", width=150)

st.sidebar.title("Settings")

use_spacy = st.sidebar.checkbox("Use spaCy NER (if available)", value=False)
use_ml = st.sidebar.checkbox("Use ML model (trained)", value=False)
normalizer = load_normalizer(use_spacy=use_spacy, use_ml=use_ml)

st.title("Claims Description Normalizer")
st.write("Transform raw claim notes into structured JSON (loss type, severity, assets, entities, confidence).")

mode = st.radio("Mode", ["Single Claim", "Batch / Evaluation"])

if mode == "Single Claim":
    st.subheader("Single Claim")
    example = st.expander("See example claims")
    with example:
        st.markdown("- Water leak in kitchen damaged the floor. Estimate $3,500 in repairs.\n- Car rear-ended on highway, vehicle totaled.")
    text = st.text_area("Enter claim text", height=180, placeholder="Type or paste claim description here...")
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Analyze"):
            if not text.strip():
                st.error("Please enter claim text.")
            else:
                res = normalizer.normalize(text)
                st.success("Analysis complete")
                # Display JSON
                st.code(json.dumps(res.to_dict(), indent=2))
                # Show metrics
                st.metric("Confidence", f"{res.confidence_score*100:.0f}%")
                if res.affected_assets:
                    st.write("**Affected assets:**", ", ".join(res.affected_assets))
                if res.extracted_entities:
                    st.write("**Entities:**")
                    for k,v in res.extracted_entities.items():
                        st.write(f"- {k}: {', '.join(v)}")
                # Download
                st.download_button("Download JSON", data=json.dumps(res.to_dict(), indent=2).encode("utf-8"), file_name="claim_result.json")

elif mode == "Batch / Evaluation":
    st.subheader("Batch Processing & Evaluation")
    st.markdown("Upload a `.txt` (claims separated by blank lines) or a `.csv` containing at least `text` column. If CSV has `true_loss` and `true_severity` columns, evaluation metrics will be computed.")
    uploaded = st.file_uploader("Upload file", type=["txt","csv"])
    uploaded_df = None
    if uploaded:
        if uploaded.name.endswith(".txt"):
            content = uploaded.read().decode("utf-8")
            claims = [c.strip() for c in content.split("\n\n") if c.strip()]
            st.write(f"Found {len(claims)} claims in TXT file.")
            # process
            results = [normalizer.normalize(c) for c in claims]
            df = pd.DataFrame([r.to_dict() for r in results])
            uploaded_df = df
        else:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.error("CSV must include a `text` column.")
            else:
                st.write(f"CSV loaded: {len(df)} rows.")
                results = [normalizer.normalize(t) for t in df["text"].astype(str).tolist()]
                results_df = pd.DataFrame([r.to_dict() for r in results])
                uploaded_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
                # If true labels exist, evaluate
                if {"true_loss","true_severity"}.issubset(df.columns):
                    metrics = normalizer.evaluate(df["text"].astype(str).tolist(), df["true_loss"].astype(str).tolist(), df["true_severity"].astype(str).tolist())
                    st.subheader("Evaluation Metrics (Loss Type & Severity)")
                    st.write(metrics)
                    # Simple display
                    st.metric("Loss Accuracy", f"{metrics['loss']['accuracy']*100:.1f}%")
                    st.metric("Severity Accuracy", f"{metrics['severity']['accuracy']*100:.1f}%")
        if uploaded_df is not None:
            st.download_button("Download results (CSV)", data=uploaded_df.to_csv(index=False).encode("utf-8"), file_name="claims_results.csv")
            st.dataframe(uploaded_df.head(50))

# Footer
st.markdown("---")
st.caption("Claims Description Normalizer — improved UI & evaluation. Add sample dataset to train the ML model for higher accuracy.")
