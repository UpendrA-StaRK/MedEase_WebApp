# Disease Prediction and Specialist Recommendation System

## Overview
This project is a disease prediction and specialist recommendation system that utilizes machine learning techniques to predict diseases based on symptoms and recommend medical specialists and potential surgeries. Additionally, it features an interactive web interface with a modern UI design using Streamlit.

## Features
- **Disease Prediction**: Uses K-Nearest Neighbors (KNN) and Gaussian Naïve Bayes (GNB) to classify diseases based on symptoms.
- **Specialist Recommendation**: Maps predicted diseases to corresponding medical specialists.
- **Surgery Type Prediction**: Uses a TF-IDF model and KNN to suggest doctors for surgeries.
- **Interactive Web Interface**: Implemented using Streamlit with a glass morphism UI design.
- **Chatbot Integration**: Kommunicate chatbot for user assistance.

## Installation
### Prerequisites
Ensure you have Python installed (version 3.x recommended).

### Required Libraries
Install the required dependencies using the following command:
```sh
pip install numpy pandas scikit-learn streamlit
```

## Dataset Preparation
The system utilizes the following CSV files:

1. **Original_Dataset.csv** - Contains information about symptoms and corresponding diseases.

   | Symptom_1 | Symptom_2 | Symptom_3 | ... | Symptom_N | Disease   |
   |-----------|-----------|-----------|-----|-----------|-----------|
   | Fever     | Cough     | Headache  | ... | Fatigue   | Influenza |
   | Rash      | Itching   | Swelling  | ... | Redness   | Allergy   |

2. **Doctor_Versus_Disease.csv** - Maps diseases to specialists.

   | Disease  | Specialist         |
   |----------|--------------------|
   | Influenza | General Physician |
   | Allergy  | Dermatologist     |
   | Pneumonia | Pulmonologist    |

3. **SurgerySpecialist.csv** - Contains information about available doctors for different surgery types.

   | Doctor Name  | Specialization  | Hospital       | Contact     |
   |-------------|----------------|---------------|------------|
   | Dr. John Doe | Cardiologist   | XYZ Hospital  | 1234567890 |
   | Dr. Jane Smith | Neurologist  | ABC Clinic    | 0987654321 |

4. **medical_specialist_counts.csv** - Provides statistical data on the availability of medical specialists.

   | Specialist         | Count |
   |--------------------|-------|
   | General Physician | 120   |
   | Cardiologist      | 80    |

## Usage
### Running the System
1. Load the dataset: The script reads `Original_Dataset.csv` and preprocesses it.
2. Preprocess symptoms: Symptoms are extracted, and a binary representation is created for machine learning.
3. Train the disease prediction model:
   - KNN is used to predict diseases based on symptoms.
   - Gaussian Naïve Bayes (GNB) is trained to classify diseases.
4. Predict the disease:
   - User inputs symptoms via the web interface.
   - The trained GNB model predicts the most likely disease.
5. Recommend a specialist: The script maps the predicted disease to a medical specialist using `Doctor_Versus_Disease.csv`.
6. Find doctors for the recommended surgery:
   - Uses TF-IDF vectorization to find similar surgery types.
   - Uses KNN to find the nearest qualified doctors.
7. Display results: The system outputs the predicted disease, recommended specialist, and a list of doctors.

### Running the Script
Execute the script using:
```sh
python main.py
```

## Explanation of Core Components
### 1. Machine Learning Models Used
- **K-Nearest Neighbors (KNN) for Disease Prediction**
  - KNN is used to classify diseases based on symptom similarity.
  - It finds the k nearest data points and determines the disease based on the majority class.
- **Gaussian Naïve Bayes (GNB) for Disease Prediction**
  - GNB is trained on the dataset to classify diseases based on given symptoms.
  - It assumes that symptoms are independent given the disease and uses probability distributions for prediction.
- **TF-IDF with Sigmoid Kernel for Surgery and Doctor Recommendation**
  - TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert surgery types into numerical vectors.
  - Sigmoid kernel similarity is applied to find the most similar surgeries for a given input.
- **K-Nearest Neighbors (KNN) for Finding Similar Doctors**
  - Another KNN model is trained to find doctors based on surgery type and qualifications.
  - It suggests the top 10 closest matches for a given surgery type.

### 2. Specialist Recommendation
- Maps the predicted disease to a specialist using a dictionary derived from `Doctor_Versus_Disease.csv`.

### 3. Surgery and Doctor Recommendation
- Uses TF-IDF vectorization with sigmoid kernel similarity to find related surgeries.
- Uses KNN to recommend the top 10 doctors for a given surgery.

## Example Usage
User inputs symptoms in a comma-separated format, e.g.:
```sh
fever,cough,headache
```
The script outputs:
```
Predicted Disease: Influenza
Recommended Specialist: General Physician
Recommended Surgery: [Flu Treatment]
Recommended Doctors: Dr. John Doe, Dr. Jane Smith, ...
```

## Streamlit Web Interface
The system features an interactive web interface with a glass morphism design.

- **UI Enhancements**: Custom CSS for glass morphism effects, form styling, and interactive elements.
- **User Input**: A sidebar allows users to select symptoms via a multi-select dropdown.
- **Diagnosis Results**: Displays the predicted disease, specialist recommendation, and confidence metrics.
- **Doctor Recommendations**: Provides top specialist matches and alternative doctors.

## Chatbot Integration
The system includes a **Kommunicate AI chatbot** for user assistance:
```html
<script type="text/javascript">
    (function(d, m){
        var kommunicateSettings = {
            "appId": "btech-r9w2x",
            "automaticChatOpenOnNavigation": true,
            "popupWidget": true
        };
        var s = document.createElement("script");
        s.type = "text/javascript";
        s.async = true;
        s.src = "https://widget.kommunicate.io/v2/kommunicate.app";
        var h = document.getElementsByTagName("head")[0];
        h.appendChild(s);
        window.kommunicate = m;
        m._globals = kommunicateSettings;
    })(document, window.kommunicate || {});
</script>
```

## Future Improvements
- Implement a **graphical user interface** using more advanced frameworks.
- Improve accuracy by integrating **deep learning models**.
- Expand the database with **more symptoms, diseases, and specialists**.

---

**Crafted by Upendra, Driven by Machine Learning 🤖**

