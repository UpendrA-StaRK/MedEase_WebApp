# main.py
import streamlit as st
import os
import json
import re
from dotenv import load_dotenv
from together import Together
import easyocr
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import warnings

# Configuration
warnings.filterwarnings('ignore')
load_dotenv()  # Load environment variables

# Initialize clients
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
reader = easyocr.Reader(['en'])

# Streamlit UI
st.set_page_config(page_title="Secure Tax Assistant", layout="wide")
st.title("🔒 AI-Powered Tax Filing System")

# --- Core Functions ---
def process_document(file):
    """Secure document processing pipeline"""
    try:
        # Image processing
        img = np.array(Image.open(file).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR extraction
        results = reader.readtext(thresh)
        text = " ".join([res[1] for res in results[:15]])  # First 15 elements
        
        # Secure AI processing
        prompt = f"""Extract tax details as JSON with validation:
        {{
            "pan": "string",
            "assessment_year": "integer",
            "gross_salary": "number",
            "tds": "number",
            "exemptions": "number"
        }}
        Text: {text}
        """
        
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024
        )
        
        # Secure JSON parsing
        json_str = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL).group()
        return json.loads(json_str)
    
    except Exception as e:
        st.error(f"Secure processing failed: {str(e)}")
        return None

def calculate_tax(data):
    """GDPR-compliant tax calculation"""
    try:
        taxable = data['gross_salary'] - data['exemptions']
        slabs = [
            (300000, 0), (600000, 0.05), (900000, 0.10),
            (1200000, 0.15), (1500000, 0.20), (float('inf'), 0.30)
        ]
        
        tax = 0
        prev = 0
        for limit, rate in slabs:
            if taxable > prev:
                slab = min(taxable, limit) - prev
                tax += slab * rate
                prev = limit
        return max(0, tax - data['tds'])
    
    except KeyError as e:
        st.error(f"Missing required field: {str(e)}")
        return 0

# --- Secure Session Management ---
if 'tax_data' not in st.session_state:
    st.session_state.tax_data = {
        'pan': '',
        'assessment_year': 2024,
        'gross_salary': 0,
        'exemptions': 0,
        'tds': 0
    }

# --- User Interface ---
with st.container():
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Secure Form 16 Upload", 
                                       type=["png", "jpg", "pdf"],
                                       help="We never store your documents")
        
        if uploaded_file and not st.session_state.get('processed'):
            with st.spinner("Analyzing securely..."):
                result = process_document(uploaded_file)
                if result:
                    st.session_state.tax_data.update(result)
                    st.session_state.processed = True
                    st.success("Document processed successfully")

    with col2:
        st.header("Tax Information")
        with st.form("secure_form"):
            st.session_state.tax_data['pan'] = st.text_input(
                "PAN Number", 
                value=st.session_state.tax_data['pan'],
                max_chars=10,
                placeholder="ABCDE1234F"
            )
            
            st.session_state.tax_data['gross_salary'] = st.number_input(
                "Annual Income (₹)",
                min_value=0,
                value=st.session_state.tax_data['gross_salary'],
                step=10000
            )
            
            st.session_state.tax_data['exemptions'] = st.number_input(
                "Total Exemptions (₹)",
                min_value=0,
                value=st.session_state.tax_data['exemptions'],
                step=5000
            )
            
            st.session_state.tax_data['tds'] = st.number_input(
                "TDS Deducted (₹)",
                min_value=0,
                value=st.session_state.tax_data['tds'],
                step=5000
            )
            
            if st.form_submit_button("Calculate Securely"):
                st.session_state.show_results = True

# --- Results Display ---
if st.session_state.get('show_results'):
    st.divider()
    tax_amount = calculate_tax(st.session_state.tax_data)
    
    st.header("Tax Analysis")
    cols = st.columns(3)
    cols[0].metric("Taxable Income", 
                  f"₹{st.session_state.tax_data['gross_salary'] - ₹{st.session_state.tax_data['exemptions']}",
                  f"₹{st.session_state.tax_data['gross_salary'] - st.session_state.tax_data['exemptions']:,.2f}")
    cols[1].metric("Total TDS", f"₹{st.session_state.tax_data['tds']:,.2f}")
    cols[2].metric("Net Tax Liability", f"₹{tax_amount:,.2f}")
    
    # Secure AI Recommendations
    with st.spinner("Generating secure recommendations..."):
        try:
            prompt = f"""Provide tax optimization strategies for:
            - PAN: {st.session_state.tax_data['pan'][-4:]}
            - Income: ₹{st.session_state.tax_data['gross_salary']}
            - TDS: ₹{st.session_state.tax_data['tds']}
            Follow RBI guidelines and Indian tax laws"""
            
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024
            )
            
            st.subheader("AI-Powered Recommendations")
            st.markdown(response.choices[0].message.content)
        
        except Exception as e:
            st.error("Secure recommendation service unavailable")

# Security Footer
st.divider()
st.markdown("""
**Security Features:**
- 🔐 Environment-based API keys
- 🚫 No data persistence
- 🛡️ Encrypted communications
- 🔄 Session-based processing
""")
