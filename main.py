import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

# ---------------------- Original Code (No Changes) ----------------------
disease_predict_df=pd.read_csv('Original_Dataset.csv')
disease_predict_df.fillna(0,inplace=True)
column_values =disease_predict_df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4','Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9','Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14','Symptom_15', 'Symptom_16', 'Symptom_17']].values.ravel()
symps = pd.unique(column_values).tolist()
symp = [i for i in symps if str(i) != "0"]
disease_symptom_df = pd.DataFrame(columns=['Disease'] + symp)
disease_symptom_df['Disease']=disease_predict_df['Disease']
disease_predict_df["symptoms"] = [[] for _ in range(len(disease_predict_df))]
for i in range(len(disease_predict_df)):
    row_values = disease_predict_df.iloc[i].values.tolist()
    if 0 in row_values:
        symptoms_list = row_values[1:row_values.index(0)]
    else:
        symptoms_list = row_values[1:]
    disease_predict_df.at[i, "symptoms"] = symptoms_list
symptoms_series = pd.Series(disease_predict_df["symptoms"] )
disease_symptom_df.iloc[:, 1:] = 0
for i in range(len(disease_symptom_df)):
    symptoms = symptoms_series.iloc[i]
    for symptom in symptoms:
        if symptom in symp:
            disease_symptom_df.at[i, symptom] = 1
X = disease_symptom_df.drop(columns=['Disease'])
y = disease_symptom_df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
symp_model_disease = KNeighborsClassifier()
symp_model_disease.fit(X_train, y_train)
y_pred = symp_model_disease.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

doc_vs_disease = pd.read_csv('Doctor_Versus_Disease.csv', header=None, index_col=None, encoding='ISO-8859-1')
dictionary = doc_vs_disease.set_index(0).to_dict()[1]
def predict_specialist(predicted_disease):
    return dictionary.get(predicted_disease, 'Unknown Specialist')
X = disease_symptom_df.drop(columns=['Disease'])
y = disease_symptom_df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
disease_model_surgery = GaussianNB()
disease_model_surgery.fit(X_train, y_train)
y_pred = disease_model_surgery.predict(X_test)
specialist_recommendations = [predict_specialist(disease) for disease in y_pred]
comparison_df = pd.DataFrame({
    'Actual': y_test.reset_index(drop=True).head(len(y_pred)).values,
    'Predicted': y_pred,
    'Specialist_Recommended': [predict_specialist(disease) for disease in y_pred]
})
accuracy = accuracy_score(y_test, y_pred)

doctors=pd.read_csv('SurgerySpecialist.csv')
specialist_count=pd.read_csv('medical_specialist_counts.csv')
unique_surgery_types = doctors['SURGERY TYPE'].unique()
def normalize(s):
    return s.strip().lower()
normalized_surgery_set = set(normalize(s) for s in unique_surgery_types)
normalized_dict_set = set(normalize(s) for s in dictionary.values())
matching_specialists = normalized_surgery_set.intersection(normalized_dict_set)
Unique_Disease=doctors['Medical Intervention'].unique()
def normalize(s):
    return s.strip().lower()
normalized_surgery_set = set(normalize(s) for s in Unique_Disease)
normalized_dict_set = set(normalize(s) for s in dictionary.keys())
matching_specialists = normalized_surgery_set.intersection(normalized_dict_set)
specific_values = ['Dermatologist', 'Gynecologist', 'Gastroenterologist', 'Cardiologist']
filtered_df = doctors[doctors['SURGERY TYPE'].isin(specific_values)]
doctors=filtered_df
final_df=doctors
final_df['Medical Intervention'] = final_df['Medical Intervention'].fillna('')
final_df=final_df.dropna(axis=0)
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',ngram_range=(1, 3), stop_words='english')
tfv_matrix = tfv.fit_transform(final_df['SURGERY TYPE'])
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(final_df.index, index=final_df['SURGERY TYPE']).drop_duplicates()
def model1(surgery, sig=sig):
    idx = indices[surgery]
    return list(dict.fromkeys(final_df[final_df['SURGERY TYPE']==surgery]['Name']))[:10]
le_surgery = LabelEncoder()
final_df['SURGERY TYPE Encoded'] = le_surgery.fit_transform(final_df['SURGERY TYPE'])
final_df['QUALIFICATIONS Encoded'] = final_df['QUALIFICATIONS'].astype('category').cat.codes
features = final_df[['SURGERY TYPE Encoded', 'QUALIFICATIONS Encoded']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
knn = NearestNeighbors(n_neighbors=10, algorithm='auto')
knn.fit(features_scaled)
def model2(surgery_name):
    idx = final_df[final_df['SURGERY TYPE'] == surgery_name].index[0]
    surgery_features = features_scaled[idx].reshape(1, -1)
    distances, indices = knn.kneighbors(surgery_features)
    similar_indices = indices[0][1:]
    return list(dict.fromkeys(final_df['Name'].iloc[similar_indices]))[:10]
referene_df = pd.DataFrame(columns=symp)
listings = []
for i in range(len(symp)):
    listings.append(0)
referene_df.loc[0] = listings

# ---------------------- Updated Streamlit UI with Glass Morphism ----------------------
st.set_page_config(page_title="MedEase", page_icon="🌊", layout="wide")

st.markdown("""
<style>
/* Main background gradient */
.stApp {
  background: 
    linear-gradient(150deg, #21ff3a, #394e41, #17172c, #2a81ff) !important;
  background-size: cover;
}

/* Glass morphism effects */
.glass-container {
    background: rgb(0 0 0 / 50%) !important ;
    backdrop-filter: blur(12px) !important;
    border-radius: 15px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15) !important;
    padding: 25px !important;
    margin: 15px 0 !important;
    color: white !important;
}

/* Header styling */
.header-title {
    font-size: 2.8em !important;
    font-weight: bold !important;
    color: white !important;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    padding: 20px !important;
}

/* Form elements */
.stSelectbox, .stMultiselect, .stButton>button {
    background: rgba(255, 255, 255, 0.2) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    color: white !important;
    border-radius: 10px !important;
    backdrop-filter: blur(5px) !important;
}

/* Text elements */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p {
    color: white !important;
}

/* Warning boxes */
.warning-box {
    background: rgba(255, 50, 50, 0.15) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown('<div class="header-title">🩺 MedEase - AI Health Consultant ⚕️</div>', unsafe_allow_html=True)
st.markdown("""
<div class="glass-container" style="text-align: center; margin-bottom: 30px">
    Your Intelligent Healthcare Diagnosis System with Glass Morphism Design
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🩺 Symptom Input")
    selected_symptoms = st.multiselect(
        "Select your symptoms:",
        options=symp,
        help="Start typing or select from dropdown",
        max_selections=5
    )
    
    analyze_btn = st.button("🔍 Start Diagnosis", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

# Diagnosis results
if analyze_btn:
    if len(selected_symptoms) == 0:
        st.markdown("""
        <div class="warning-box glass-container">
            ⚠️ Please select at least one symptom
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("🔬 Analyzing symptoms..."):
            # Prepare input vector
            input_vector = referene_df.copy()
            for symptom in selected_symptoms:
                if symptom in input_vector.columns:
                    input_vector.at[0, symptom] = 1
            
            # Get predictions
            disease_prediction = disease_model_surgery.predict(input_vector)[0]
            specialist = dictionary.get(disease_prediction, 'General Physician')
            
            # Get recommendations
            try: model1_doctors = model1(specialist)
            except: model1_doctors = []
            try: model2_doctors = model2(specialist)
            except: model2_doctors = []

            # Display results
            st.markdown(f"""
            <div class="glass-container">
                <h2>📋 Diagnosis Report</h2>
                <div style="padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px">
                    <p style="font-size: 1.2em">🧬 Predicted Condition: <strong>{disease_prediction}</strong></p>
                    <p style="font-size: 1.2em">⚕️ Recommended Specialist: <strong>{specialist}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Accuracy metrics
            st.markdown(f"""
            <div class="glass-container">
                <h3>📈 Confidence Metrics</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px">
                    <div style="padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px">
                        <p>🩺 Symptom Recognition</p>
                        <h2 style="color: #00ff00">{accuracy*100:.1f}%</h2>
                    </div>
                    <div style="padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px">
                        <p>🏥 Specialist Matching</p>
                        <h2 style="color: #00ff00">{accuracy*100:.1f}%</h2>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Doctor recommendations
            st.markdown("""
            <div class="glass-container">
                <h3>⚕️ Recommended Specialists</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px">
            """, unsafe_allow_html=True)
            
            cols = st.columns(2)
            with cols[0]:
                st.markdown("""
                <div style="padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px">
                    <h4>🏅 Top Matches</h4>
                """, unsafe_allow_html=True)
                for doc in model1_doctors:
                    st.markdown(f"- ⚕️ {doc}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown("""
                <div style="padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px">
                    <h4>🌟 Alternatives</h4>
                """, unsafe_allow_html=True)
                for doc in model2_doctors:
                    st.markdown(f"- ⚕️ {doc}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="glass-container" style="margin-top: 30px">
    📢 Note: This AI system provides preliminary health insights and should not replace professional medical advice. 
    Always consult a certified healthcare provider for diagnosis and treatment.
</div>
""", unsafe_allow_html=True)

# Bottom text
st.markdown(
    """
    <style>
        .bottom-text {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 1.5em;
            font-weight: bold;
        }
    </style>
    <div class="bottom-text">🚀 Crafted by Upendra, Driven by Machine Learning 🤖</div>
    """,
    unsafe_allow_html=True
)
