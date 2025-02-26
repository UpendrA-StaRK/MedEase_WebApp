import streamlit as st
import os
from dotenv import load_dotenv
import json
from together import Together
import easyocr
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
together.api_key = os.getenv("442ba1c799d5df6df52c20a6ea6970d7af90b32e5c31199689e5e676fc6e1f0e")
client = Together(api_key=os.getenv("442ba1c799d5df6df52c20a6ea6970d7af90b32e5c31199689e5e676fc6e1f0e"))

# Initialize OCR reader
reader = easyocr.Reader(['en'])

st.title("Automated Tax Filing Assistant")

# ... (Keep all expander sections identical from original code) ...

# --- File Upload Section ---
st.header("📁 Upload Form 16")
uploaded_file = st.file_uploader("Upload Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

# --- Function to Process Image (Updated AI Integration) ---
def process_image(file):
    """Extracts text from image using OCR and processes it into structured JSON."""
    try:
        # Convert image for OpenCV
        image = Image.open(file).convert("RGB")
        image = np.array(image)

        # Convert to grayscale & apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Extract text using OCR
        results = reader.readtext(thresh)
        extracted_text = " ".join([text[1] for text in results])

        if not extracted_text.strip():
            st.error("No readable text found. Try a clearer image.")
            return {}

        # AI Prompt for structured JSON extraction using new API
        prompt = f"""
        Extract the following fields in valid JSON:
        - pan (string)
        - assessment_year (integer)
        - employment_from (YYYY-MM-DD)
        - employment_to (YYYY-MM-DD)
        - gross_salary (number)
        - exemptions (number)
        - section16_deductions (number)
        - other_income (number)
        - chapter6_deductions (number)
        - tds (number)

        Text: {extracted_text[:3000]}
        Output only the JSON object.
        """

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048
        )

        json_str = response.choices[0].message.content.strip()
        
        # Extract and clean JSON
        import re
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            st.error("Invalid JSON from AI.")
            return {}

        return json.loads(json_str)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# --- Tax Form Auto-Fill with Session State ---
if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'pan': "",
        'assessment_year': 2024,
        'employment_from': datetime(2023, 4, 1),
        'employment_to': datetime(2024, 3, 31),
        'gross_salary': 0,
        'exemptions': 0,
        'section16_deductions': 0,
        'other_income': 0,
        'chapter6_deductions': 0,
        'tds': 0
    }

# Auto-fill form fields if image uploaded
if uploaded_file and 'image_processed' not in st.session_state:
    with st.spinner("Analyzing document..."):
        extracted_data = process_image(uploaded_file)
        if extracted_data:
            for key in st.session_state.form_data.keys():
                if key in extracted_data and extracted_data[key] not in [None, ""]:
                    st.session_state.form_data[key] = extracted_data[key]
            st.session_state.image_processed = True

# Create form with session state values
with st.form("tax_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.form_data['pan'] = st.text_input("PAN Number", 
            value=st.session_state.form_data['pan'], max_chars=10)
        st.session_state.form_data['assessment_year'] = st.selectbox("Assessment Year", 
            [2024, 2023, 2022], index=[2024, 2023, 2022].index(st.session_state.form_data['assessment_year']))
        st.session_state.form_data['employment_from'] = st.date_input("Employment Start", 
            value=st.session_state.form_data['employment_from'])
        st.session_state.form_data['gross_salary'] = st.number_input("Gross Salary (₹)", 
            min_value=0, step=10000, value=st.session_state.form_data['gross_salary'])
        
    with col2:
        st.session_state.form_data['employment_to'] = st.date_input("Employment End", 
            value=st.session_state.form_data['employment_to'])
        st.session_state.form_data['exemptions'] = st.number_input("Total Exemptions (₹)", 
            min_value=0, step=1000, value=st.session_state.form_data['exemptions'])
        st.session_state.form_data['tds'] = st.number_input("TDS Deducted (₹)", 
            min_value=0, step=1000, value=st.session_state.form_data['tds'])
    
    st.session_state.form_data['other_income'] = st.number_input("Other Income (₹)", 
        min_value=0, step=10000, value=st.session_state.form_data['other_income'])
    st.session_state.form_data['section16_deductions'] = st.number_input("Section 16 Deductions (₹)", 
        min_value=0, step=5000, value=st.session_state.form_data['section16_deductions'])
    st.session_state.form_data['chapter6_deductions'] = st.number_input("Chapter VI-A Deductions (₹)", 
        min_value=0, step=5000, value=st.session_state.form_data['chapter6_deductions'])
    
    submitted = st.form_submit_button("Calculate Tax Liability")

# --- Enhanced Tax Calculation ---
def calculate_tax(data):
    """Calculates tax liability as per FY 2023-24 with detailed breakdown."""
    gross_income = data['gross_salary'] + data['other_income']
    exempt_income = data['exemptions']
    taxable_income = gross_income - exempt_income
    deductions = data['section16_deductions'] + data['chapter6_deductions']
    net_taxable = max(0, taxable_income - deductions)

    # Tax slabs
    tax = 0
    previous_limit = 0
    tax_slabs = [
        (300000, 0),
        (600000, 0.05),
        (900000, 0.10),
        (1200000, 0.15),
        (1500000, 0.20),
        (float('inf'), 0.30)
    ]

    for limit, rate in tax_slabs:
        if net_taxable > previous_limit:
            current_slab = min(net_taxable, limit) - previous_limit
            tax += current_slab * rate
            previous_limit = limit
        else:
            break

    tax_payable = max(0, tax - data['tds'])
    
    return {
        'gross_income': gross_income,
        'taxable_income': taxable_income,
        'deductions': deductions,
        'net_taxable': net_taxable,
        'total_tax': tax,
        'tds': data['tds'],
        'tax_payable': tax_payable
    }

# --- Enhanced AI Tax Advice ---
def get_ai_advice(data):
    """Generates tax-saving recommendations using latest LLM."""
    try:
        prompt = f"""
        Suggest 5 strategies to reduce tax liability for an Indian taxpayer with:
        - Gross Income: ₹{data['gross_salary']}
        - Chapter VI-A Deductions: ₹{data['chapter6_deductions']}
        - TDS Deducted: ₹{data['tds']}

        Provide section numbers and calculation examples according to Indian tax laws.
        """
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"AI service error: {str(e)}"

# --- Display Enhanced Results ---
if submitted:
    st.header("📊 Tax Analysis")
    tax_results = calculate_tax(st.session_state.form_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Gross Income", value=f"₹ {tax_results['gross_income']:,.2f}")
        st.metric(label="Total Deductions", value=f"₹ {tax_results['deductions']:,.2f}")
    with col2:
        st.metric(label="Taxable Income", value=f"₹ {tax_results['taxable_income']:,.2f}")
        st.metric(label="TDS Deducted", value=f"₹ {tax_results['tds']:,.2f}")
    with col3:
        st.metric(label="Total Tax", value=f"₹ {tax_results['total_tax']:,.2f}")
        st.metric(label="Net Tax Payable", value=f"₹ {tax_results['tax_payable']:,.2f}")

    st.subheader("🧠 AI Recommendations")
    with st.spinner("Generating strategies..."):
        advice = get_ai_advice(st.session_state.form_data)
        st.markdown(advice)

st.markdown("---")
st.markdown("🔹 **Disclaimer**: Consult a CA for official tax filing.")
