import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

# Change Name & Logo
st.set_page_config(
    page_title="Medical AI Diagnosis", 
    page_icon="⚕️",
    layout="wide"
)

# Hiding Streamlit add-ons
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Custom CSS for improved appearance
custom_css = """
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #4FC3F7 !important;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #B2EBF2 !important;
        margin-bottom: 2rem;
        text-align: center;
    }
    .section-header {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #81D4FA !important;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: rgba(13, 71, 161, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .positive-result {
        background-color: rgba(217, 48, 37, 0.8);
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }
    .negative-result {
        background-color: rgba(46, 125, 50, 0.8);
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }
    .result-container {
        margin-top: 30px;
        font-size: 1.2rem;
    }
    .form-container {
        background-color: rgba(25, 42, 86, 0.7);
        padding: 20px;
        border-radius: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Adding Background Image
background_image_url = "https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg"  # Replace with your image URL

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url({background_image_url});
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stAppViewContainer"]::before {{
content: "";
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
background-color: rgba(0, 0, 0, 0.75);
}}

[data-testid="stSidebar"] {{
background-color: rgba(10, 25, 41, 0.8);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Header Section
st.markdown('<p class="main-header">Medical AI Diagnosis System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced disease prediction using machine learning</p>', unsafe_allow_html=True)

# Load the saved models
models = {
    'diabetes': pickle.load(open('Models/diabetes_model.sav', 'rb')),
    'heart_disease': pickle.load(open('Models/heart_disease_model.sav', 'rb')),
    'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
    'lung_cancer': pickle.load(open('Models/lungs_disease_model.sav', 'rb')),
    'thyroid': pickle.load(open('Models/Thyroid_model.sav', 'rb'))
}

# Create a sidebar menu for disease prediction
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4807/4807695.png", width=100)
    st.title("Navigation")
    selected = option_menu(
        menu_title=None,
        options=['Home', 'Diabetes', 'Heart Disease', 'Parkinson\'s', 'Lung Cancer', 'Hypo-Thyroid'],
        icons=['house', 'activity', 'heart', 'person', 'lungs', 'radioactive'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "rgba(10, 25, 41, 0.8)"},
            "icon": {"color": "#4FC3F7", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#4A4A4A"},
            "nav-link-selected": {"background-color": "#1E88E5"},
        }
    )

def display_input(label, tooltip, key, min_val=None, max_val=None, type="text", options=None, default=None):
    """Enhanced input display function with better tooltips and validation"""
    if type == "text":
        return st.text_input(label, key=key, help=tooltip, value=default)
    elif type == "number":
        if min_val is not None and max_val is not None:
            return st.number_input(label, key=key, help=tooltip, min_value=min_val, max_value=max_val, value=default if default is not None else min_val)
        else:
            return st.number_input(label, key=key, help=tooltip, value=default if default is not None else 0)
    elif type == "slider":
        # Ensure all slider values are the same type to avoid Streamlit errors
        if isinstance(min_val, float) or isinstance(max_val, float):
            # Convert all to float if any value is float
            min_val_float = float(min_val)
            max_val_float = float(max_val)
            default_val = float(default) if default is not None else min_val_float
            step = 0.1  # Use appropriate step for float values
            return st.slider(label, key=key, help=tooltip, min_value=min_val_float, max_value=max_val_float, 
                           value=default_val, step=step)
        else:
            # All values are integers
            return st.slider(label, key=key, help=tooltip, min_value=min_val, max_value=max_val, 
                           value=default if default is not None else min_val)
    elif type == "select":
        return st.selectbox(label, options, index=0 if default is None else options.index(default), help=tooltip)

def display_disease_info(title, description, symptoms, risk_factors):
    """Display formatted disease information"""
    st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)
    
    with st.expander("About this disease", expanded=False):
        st.markdown(f"""
        <div class="info-box">
            <h4>Description</h4>
            <p>{description}</p>
            <h4>Common Symptoms</h4>
            <p>{symptoms}</p>
            <h4>Risk Factors</h4>
            <p>{risk_factors}</p>
        </div>
        """, unsafe_allow_html=True)

def display_result(prediction, positive_message, negative_message):
    """Display formatted prediction result"""
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(f'<div class="positive-result">{positive_message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="negative-result">{negative_message}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Home Page
if selected == 'Home':
    st.markdown('<p class="section-header">Welcome to Medical AI Diagnosis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>About This Application</h3>
        <p>This application uses machine learning algorithms to predict the likelihood of five common medical conditions based on patient data. The models have been trained on medical datasets and can provide preliminary assessments.</p>
        
        <h3>Available Disease Predictions</h3>
        <ul>
            <li><strong>Diabetes</strong>: Predicts diabetes based on medical and demographic factors</li>
            <li><strong>Heart Disease</strong>: Evaluates the risk of coronary heart disease</li>
            <li><strong>Parkinson's Disease</strong>: Analyzes voice recordings for Parkinson's indicators</li>
            <li><strong>Lung Cancer</strong>: Assesses lung cancer risk based on symptoms and history</li>
            <li><strong>Hypo-Thyroid</strong>: Evaluates thyroid function and detects hypothyroidism</li>
        </ul>
        
        <h3>How to Use</h3>
        <p>Select a disease from the sidebar menu, fill in the required information, and click the prediction button to get your result.</p>
        
        <h3>Disclaimer</h3>
        <p>This tool is for educational purposes only and does not replace professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.</p>
    </div>
    """, unsafe_allow_html=True)

# Diabetes Prediction Page
elif selected == 'Diabetes':
    display_disease_info(
        "Diabetes Prediction",
        "Diabetes is a chronic disease that occurs when the pancreas is no longer able to make insulin, or when the body cannot make good use of the insulin it produces.",
        "Frequent urination, increased thirst, extreme hunger, unexplained weight loss, fatigue, irritability, blurred vision, slow-healing sores.",
        "Family history, age, excess weight, physical inactivity, race, high blood pressure, abnormal cholesterol levels."
    )
    
    with st.form("diabetes_form"):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("Enter Patient Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            Pregnancies = display_input('Number of Pregnancies', 'Number of times pregnant', 'Pregnancies', 0, 20, 'number')
            Glucose = display_input('Glucose Level', 'Plasma glucose concentration (mg/dL)', 'Glucose', 0.0, 200.0, 'slider')
            BloodPressure = display_input('Blood Pressure', 'Diastolic blood pressure (mm Hg)', 'BloodPressure', 0.0, 122.0, 'slider')
            SkinThickness = display_input('Skin Thickness', 'Triceps skin fold thickness (mm)', 'SkinThickness', 0.0, 100.0, 'slider')
        
        with col2:
            Insulin = display_input('Insulin Level', '2-Hour serum insulin (mu U/ml)', 'Insulin', 0, 846, 'number')
            BMI = display_input('BMI value', 'Body Mass Index (weight in kg/(height in m)²)', 'BMI', 0.0, 67.1, 'slider')
            DiabetesPedigreeFunction = display_input('Diabetes Pedigree Function', 'Diabetes pedigree function (genetic influence)', 'DiabetesPedigreeFunction', 0.078, 2.42, 'slider')
            Age = display_input('Age', 'Age in years', 'Age', 21.0, 81.0, 'slider')
        
        # Submit buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            positive_sample = st.form_submit_button("Load & Predict with Positive Sample")
        with col2:
            negative_sample = st.form_submit_button("Load & Predict with Negative Sample")
        with col3:
            submitted = st.form_submit_button("Predict with Current Data")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Handle predictions with sample data or user input
    if positive_sample:
        # Load positive sample values
        Pregnancies = 6
        Glucose = 148
        BloodPressure = 72
        SkinThickness = 35
        Insulin = 155 
        BMI = 33.6
        DiabetesPedigreeFunction = 0.627
        Age = 50
        
        # Make prediction with positive sample
        try:
            diab_prediction = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            display_result(
                diab_prediction[0], 
                "POSITIVE: The analysis indicates diabetes. Please consult with a healthcare provider.",
                "NEGATIVE: The analysis does not indicate diabetes. Maintain a healthy lifestyle."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    elif negative_sample:
        # Load negative sample values
        Pregnancies = 1
        Glucose = 85
        BloodPressure = 66
        SkinThickness = 29
        Insulin = 85
        BMI = 26.6
        DiabetesPedigreeFunction = 0.351
        Age = 31
        
        # Make prediction with negative sample
        try:
            diab_prediction = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            display_result(
                diab_prediction[0], 
                "POSITIVE: The analysis indicates diabetes. Please consult with a healthcare provider.",
                "NEGATIVE: The analysis does not indicate diabetes. Maintain a healthy lifestyle."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    elif submitted:
        try:
            diab_prediction = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            display_result(
                diab_prediction[0], 
                "POSITIVE: The analysis indicates diabetes. Please consult with a healthcare provider.",
                "NEGATIVE: The analysis does not indicate diabetes. Maintain a healthy lifestyle."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Heart Disease Prediction Page
elif selected == 'Heart Disease':
    display_disease_info(
        "Heart Disease Prediction",
        "Heart disease refers to a range of conditions that affect your heart, including coronary artery disease, heart rhythm problems (arrhythmias), and congenital heart defects.",
        "Chest pain, shortness of breath, pain in the neck, jaw, throat, upper abdomen or back, numbness, weakness, coldness in legs or arms.",
        "Age, sex, family history, smoking, poor diet, high blood pressure, high blood cholesterol, diabetes, obesity, physical inactivity, stress."
    )
    
    with st.form("heart_disease_form"):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("Enter Patient Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = display_input('Age', 'Age in years', 'age', 20.0, 100.0, 'slider')
            sex = display_input('Sex', 'Gender of the person', 'sex', type='select', options=['Female', 'Male'])
            sex = 1 if sex == 'Male' else 0
            cp = display_input('Chest Pain Type', 'Type of chest pain experienced', 'cp', type='select', 
                              options=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
            cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
            cp = cp_map[cp]
            trestbps = display_input('Resting Blood Pressure', 'Resting blood pressure (mm Hg)', 'trestbps', 90.0, 200.0, 'slider')
        
        with col2:
            chol = display_input('Serum Cholesterol', 'Serum cholesterol (mg/dl)', 'chol', 100.0, 600.0, 'slider')
            fbs = display_input('Fasting Blood Sugar', 'Fasting blood sugar > 120 mg/dl', 'fbs', type='select', options=['No', 'Yes'])
            fbs = 1 if fbs == 'Yes' else 0
            restecg = display_input('Resting ECG', 'Resting electrocardiographic results', 'restecg', type='select', 
                                   options=['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
            restecg_map = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
            restecg = restecg_map[restecg]
            thalach = display_input('Max Heart Rate', 'Maximum heart rate achieved', 'thalach', 70.0, 220.0, 'slider')
        
        with col3:
            exang = display_input('Exercise Induced Angina', 'Exercise induced angina', 'exang', type='select', options=['No', 'Yes'])
            exang = 1 if exang == 'Yes' else 0
            oldpeak = display_input('ST Depression', 'ST depression induced by exercise relative to rest', 'oldpeak', 0.0, 6.2, 'slider')
            slope = display_input('Slope of ST Segment', 'Slope of the peak exercise ST segment', 'slope', type='select', 
                                 options=['Upsloping', 'Flat', 'Downsloping'])
            slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
            slope = slope_map[slope]
            ca = display_input('Number of Major Vessels', 'Number of major vessels colored by fluoroscopy', 'ca', 0.0, 3.0, 'slider')
            thal = display_input('Thalassemia', 'Type of thalassemia', 'thal', type='select', 
                               options=['Normal', 'Fixed Defect', 'Reversible Defect'])
            thal_map = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}
            thal = thal_map[thal]
        
        # Submit buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            positive_sample = st.form_submit_button("Load & Predict with Positive Sample")
        with col2:
            negative_sample = st.form_submit_button("Load & Predict with Negative Sample")
        with col3:
            submitted = st.form_submit_button("Predict with Current Data")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Handle predictions with sample data or user input
    if positive_sample:
        # Sample values for a positive heart disease case
        age = 65
        sex = 1  # Male
        cp = 3   # Asymptomatic
        trestbps = 160
        chol = 280
        fbs = 0  # No
        restecg = 2  # Left Ventricular Hypertrophy
        thalach = 130
        exang = 1  # Yes
        oldpeak = 3.1
        slope = 2  # Downsloping
        ca = 2
        thal = 2  # Reversible Defect
        
        # Make prediction with positive sample
        try:
            heart_prediction = models['heart_disease'].predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            display_result(
                heart_prediction[0], 
                "POSITIVE: The analysis indicates heart disease. Please consult with a cardiologist.",
                "NEGATIVE: The analysis does not indicate heart disease. Maintain a healthy lifestyle."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    elif negative_sample:
        # Sample values for a negative heart disease case
        age = 40
        sex = 0  # Female
        cp = 1   # Atypical Angina
        trestbps = 120
        chol = 180
        fbs = 0  # No
        restecg = 0  # Normal
        thalach = 178
        exang = 0  # No
        oldpeak = 0.8
        slope = 0  # Upsloping
        ca = 0
        thal = 0  # Normal
        
        # Make prediction with negative sample
        try:
            heart_prediction = models['heart_disease'].predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            display_result(
                heart_prediction[0], 
                "POSITIVE: The analysis indicates heart disease. Please consult with a cardiologist.",
                "NEGATIVE: The analysis does not indicate heart disease. Maintain a healthy lifestyle."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    elif submitted:
        try:
            heart_prediction = models['heart_disease'].predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            display_result(
                heart_prediction[0], 
                "POSITIVE: The analysis indicates heart disease. Please consult with a cardiologist.",
                "NEGATIVE: The analysis does not indicate heart disease. Maintain a healthy lifestyle."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Parkinson's Prediction Page
elif selected == "Parkinson's":
    display_disease_info(
        "Parkinson's Disease Prediction",
        "Parkinson's disease is a progressive nervous system disorder that affects movement. Symptoms start gradually, sometimes with a barely noticeable tremor in just one hand.",
        "Tremor, slowed movement, rigid muscles, impaired posture and balance, loss of automatic movements, speech changes, writing changes.",
        "Age, heredity, sex (men are more likely to develop Parkinson's disease than women), exposure to toxins, serious head injury."
    )
    
    with st.form("parkinsons_form"):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("Enter Voice Recording Measurements")
        st.markdown("*Note: These are technical voice parameters measured from recordings. For actual diagnosis, a professional analysis is required.*")
        
        # Use tabs to organize the many parameters
        tab1, tab2, tab3 = st.tabs(["Frequency Measures", "Jitter & Shimmer", "Other Measures"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fo = display_input('MDVP:Fo(Hz)', 'Average vocal fundamental frequency', 'fo', 80, 260, 'number')
                fhi = display_input('MDVP:Fhi(Hz)', 'Maximum vocal fundamental frequency', 'fhi', 100, 600, 'number')
            with col2:
                flo = display_input('MDVP:Flo(Hz)', 'Minimum vocal fundamental frequency', 'flo', 60, 240, 'number')
                
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                Jitter_percent = display_input('MDVP:Jitter(%)', 'Frequency variation measure (percentage)', 'Jitter_percent', 0.0, 1.0, 'number')
                Jitter_Abs = display_input('MDVP:Jitter(Abs)', 'Absolute jitter in microseconds', 'Jitter_Abs', 0.0, 0.01, 'number')
                RAP = display_input('MDVP:RAP', 'Relative amplitude perturbation', 'RAP', 0.0, 0.05, 'number')
                PPQ = display_input('MDVP:PPQ', 'Five-point period perturbation quotient', 'PPQ', 0.0, 0.05, 'number')
                DDP = display_input('Jitter:DDP', 'Average absolute difference of differences of consecutive periods', 'DDP', 0.0, 0.1, 'number')
            with col2:
                Shimmer = display_input('MDVP:Shimmer', 'Amplitude variation measure', 'Shimmer', 0.0, 0.2, 'number')
                Shimmer_dB = display_input('MDVP:Shimmer(dB)', 'Shimmer in decibels', 'Shimmer_dB', 0.0, 2.0, 'number')
                APQ3 = display_input('Shimmer:APQ3', 'Three-point amplitude perturbation quotient', 'APQ3', 0.0, 0.05, 'number')
                APQ5 = display_input('Shimmer:APQ5', 'Five-point amplitude perturbation quotient', 'APQ5', 0.0, 0.1, 'number')
                APQ = display_input('MDVP:APQ', 'Amplitude perturbation quotient', 'APQ', 0.0, 0.15, 'number')
                DDA = display_input('Shimmer:DDA', 'Average absolute differences of consecutive differences', 'DDA', 0.0, 0.15, 'number')
                
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                NHR = display_input('NHR', 'Noise-to-harmonics ratio', 'NHR', 0.0, 0.5, 'number')
                HNR = display_input('HNR', 'Harmonics-to-noise ratio', 'HNR', 0.0, 35.0, 'number')
            with col2:
                RPDE = display_input('RPDE', 'Recurrence period density entropy', 'RPDE', 0.0, 1.0, 'number')
                DFA = display_input('DFA', 'Detrended fluctuation analysis', 'DFA', 0.0, 1.0, 'number')
                spread1 = display_input('Spread1', 'Nonlinear measure of fundamental frequency variation', 'spread1', -10.0, 10.0, 'number')
                spread2 = display_input('Spread2', 'Nonlinear measure of fundamental frequency variation', 'spread2', 0.0, 1.0, 'number')
                D2 = display_input('D2', 'Correlation dimension', 'D2', 0.0, 5.0, 'number')
                PPE = display_input('PPE', 'Pitch period entropy', 'PPE', 0.0, 1.0, 'number')
        
        # Submit buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            positive_sample = st.form_submit_button("Load & Predict with Positive Sample")
        with col2:
            negative_sample = st.form_submit_button("Load & Predict with Negative Sample")
        with col3:
            submitted = st.form_submit_button("Predict with Current Data")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Handle predictions with sample data or user input
    if positive_sample:
        # Sample values for a positive Parkinson's case
        fo = 119.992
        fhi = 157.302
        flo = 74.997
        Jitter_percent = 0.00662
        Jitter_Abs = 0.00004
        RAP = 0.00401
        PPQ = 0.00317
        DDP = 0.01204
        Shimmer = 0.04374
        Shimmer_dB = 0.42600
        APQ3 = 0.02182
        APQ5 = 0.03130
        APQ = 0.02971
        DDA = 0.06545
        NHR = 0.02211
        HNR = 21.033
        RPDE = 0.525867
        DFA = 0.741751
        spread1 = -6.759571
        spread2 = 0.162699
        D2 = 2.103956
        PPE = 0.210859
        
        # Make prediction with positive sample
        try:
            parkinsons_prediction = models['parkinsons'].predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
            display_result(
                parkinsons_prediction[0], 
                "POSITIVE: The analysis indicates Parkinson's disease. Please consult with a neurologist.",
                "NEGATIVE: The analysis does not indicate Parkinson's disease."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    elif negative_sample:
        # Sample values for a negative Parkinson's case
        fo = 169.571
        fhi = 193.516
        flo = 140.394
        Jitter_percent = 0.00370
        Jitter_Abs = 0.00002
        RAP = 0.00168
        PPQ = 0.00215
        DDP = 0.00503
        Shimmer = 0.01227
        Shimmer_dB = 0.10628
        APQ3 = 0.00558
        APQ5 = 0.00695
        APQ = 0.00781
        DDA = 0.01675
        NHR = 0.00458
        HNR = 26.775
        RPDE = 0.425493
        DFA = 0.580295
        spread1 = -5.288332
        spread2 = 0.122535
        D2 = 1.657522
        PPE = 0.125272
        
        # Make prediction with negative sample
        try:
            parkinsons_prediction = models['parkinsons'].predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
            display_result(
                parkinsons_prediction[0], 
                "POSITIVE: The analysis indicates Parkinson's disease. Please consult with a neurologist.",
                "NEGATIVE: The analysis does not indicate Parkinson's disease."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    elif submitted:
        try:
            parkinsons_prediction = models['parkinsons'].predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
            display_result(
                parkinsons_prediction[0], 
                "POSITIVE: The analysis indicates Parkinson's disease. Please consult with a neurologist.",
                "NEGATIVE: The analysis does not indicate Parkinson's disease."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Lung Cancer Prediction Page
elif selected == "Lung Cancer":
    display_disease_info(
        "Lung Cancer Prediction",
        "Lung cancer is a type of cancer that begins in the lungs. It is the leading cause of cancer deaths worldwide.",
        "Persistent cough, coughing up blood, chest pain, hoarseness, weight loss, shortness of breath, wheezing, weakness and fatigue.",
        "Smoking, exposure to secondhand smoke, exposure to radon gas, exposure to asbestos, family history of lung cancer."
    )
    
    with st.form("lung_cancer_form"):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("Enter Patient Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            GENDER = display_input('Gender', 'Gender of the person', 'GENDER', type='select', options=['Female', 'Male'])
            GENDER = 1 if GENDER == 'Male' else 0
            AGE = display_input('Age', 'Age in years', 'AGE', 1.0, 100.0, 'slider')
            SMOKING = display_input('Smoking', 'Does the person smoke?', 'SMOKING', type='select', options=['No', 'Yes'])
            SMOKING = 1 if SMOKING == 'Yes' else 0
            YELLOW_FINGERS = display_input('Yellow Fingers', 'Does the person have yellow fingers?', 'YELLOW_FINGERS', type='select', options=['No', 'Yes'])
            YELLOW_FINGERS = 1 if YELLOW_FINGERS == 'Yes' else 0
            ANXIETY = display_input('Anxiety', 'Does the person have anxiety?', 'ANXIETY', type='select', options=['No', 'Yes'])
            ANXIETY = 1 if ANXIETY == 'Yes' else 0
        
        with col2:
            PEER_PRESSURE = display_input('Peer Pressure', 'Is the person under peer pressure?', 'PEER_PRESSURE', type='select', options=['No', 'Yes'])
            PEER_PRESSURE = 1 if PEER_PRESSURE == 'Yes' else 0
            CHRONIC_DISEASE = display_input('Chronic Disease', 'Does the person have a chronic disease?', 'CHRONIC_DISEASE', type='select', options=['No', 'Yes'])
            CHRONIC_DISEASE = 1 if CHRONIC_DISEASE == 'Yes' else 0
            FATIGUE = display_input('Fatigue', 'Does the person experience fatigue?', 'FATIGUE', type='select', options=['No', 'Yes'])
            FATIGUE = 1 if FATIGUE == 'Yes' else 0
            ALLERGY = display_input('Allergy', 'Does the person have allergies?', 'ALLERGY', type='select', options=['No', 'Yes'])
            ALLERGY = 1 if ALLERGY == 'Yes' else 0
            WHEEZING = display_input('Wheezing', 'Does the person experience wheezing?', 'WHEEZING', type='select', options=['No', 'Yes'])
            WHEEZING = 1 if WHEEZING == 'Yes' else 0
        
        with col3:
            ALCOHOL_CONSUMING = display_input('Alcohol Consuming', 'Does the person consume alcohol?', 'ALCOHOL_CONSUMING', type='select', options=['No', 'Yes'])
            ALCOHOL_CONSUMING = 1 if ALCOHOL_CONSUMING == 'Yes' else 0
            COUGHING = display_input('Coughing', 'Does the person experience coughing?', 'COUGHING', type='select', options=['No', 'Yes'])
            COUGHING = 1 if COUGHING == 'Yes' else 0
            SHORTNESS_OF_BREATH = display_input('Shortness Of Breath', 'Does the person experience shortness of breath?', 'SHORTNESS_OF_BREATH', type='select', options=['No', 'Yes'])
            SHORTNESS_OF_BREATH = 1 if SHORTNESS_OF_BREATH == 'Yes' else 0
            SWALLOWING_DIFFICULTY = display_input('Swallowing Difficulty', 'Does the person have difficulty swallowing?', 'SWALLOWING_DIFFICULTY', type='select', options=['No', 'Yes'])
            SWALLOWING_DIFFICULTY = 1 if SWALLOWING_DIFFICULTY == 'Yes' else 0
            CHEST_PAIN = display_input('Chest Pain', 'Does the person experience chest pain?', 'CHEST_PAIN', type='select', options=['No', 'Yes'])
            CHEST_PAIN = 1 if CHEST_PAIN == 'Yes' else 0
        
        # Submit buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            positive_sample = st.form_submit_button("Load & Predict with Positive Sample")
        with col2:
            negative_sample = st.form_submit_button("Load & Predict with Negative Sample")
        with col3:
            submitted = st.form_submit_button("Predict with Current Data")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Handle predictions with sample data or user input
    if positive_sample:
        # Sample values for a positive lung cancer case
        GENDER = 1  # Male
        AGE = 65
        SMOKING = 1  # Yes
        YELLOW_FINGERS = 1  # Yes
        ANXIETY = 1  # Yes
        PEER_PRESSURE = 0  # No
        CHRONIC_DISEASE = 1  # Yes
        FATIGUE = 1  # Yes
        ALLERGY = 0  # No
        WHEEZING = 1  # Yes
        ALCOHOL_CONSUMING = 1  # Yes
        COUGHING = 1  # Yes
        SHORTNESS_OF_BREATH = 1  # Yes
        SWALLOWING_DIFFICULTY = 1  # Yes
        CHEST_PAIN = 1  # Yes
        
        # Make prediction with positive sample
        try:
            lungs_prediction = models['lung_cancer'].predict([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]])
            display_result(
                lungs_prediction[0], 
                "POSITIVE: The analysis indicates a risk of lung cancer. Please consult with an oncologist immediately.",
                "NEGATIVE: The analysis does not indicate lung cancer. Maintain a healthy lifestyle."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    elif negative_sample:
        # Sample values for a negative lung cancer case
        GENDER = 0  # Female
        AGE = 35
        SMOKING = 0  # No
        YELLOW_FINGERS = 0  # No
        ANXIETY = 0  # No
        PEER_PRESSURE = 0  # No
        CHRONIC_DISEASE = 0  # No
        FATIGUE = 0  # No
        ALLERGY = 0  # No
        WHEEZING = 0  # No
        ALCOHOL_CONSUMING = 0  # No
        COUGHING = 0  # No
        SHORTNESS_OF_BREATH = 0  # No
        SWALLOWING_DIFFICULTY = 0  # No
        CHEST_PAIN = 0  # No
        
        # Make prediction with negative sample
        try:
            lungs_prediction = models['lung_cancer'].predict([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]])
            display_result(
                lungs_prediction[0], 
                "POSITIVE: The analysis indicates a risk of lung cancer. Please consult with an oncologist immediately.",
                "NEGATIVE: The analysis does not indicate lung cancer. Maintain a healthy lifestyle."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    elif submitted:
        try:
            lungs_prediction = models['lung_cancer'].predict([[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]])
            display_result(
                lungs_prediction[0], 
                "POSITIVE: The analysis indicates a risk of lung cancer. Please consult with an oncologist immediately.",
                "NEGATIVE: The analysis does not indicate lung cancer. Maintain a healthy lifestyle."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Hypo-Thyroid Prediction Page
elif selected == "Hypo-Thyroid":
    display_disease_info(
        "Hypo-Thyroid Prediction",
        "Hypothyroidism is a condition in which the thyroid gland doesn't produce enough thyroid hormone. It can cause various health problems if left untreated.",
        "Fatigue, increased sensitivity to cold, constipation, dry skin, weight gain, puffy face, hoarseness, muscle weakness, elevated blood cholesterol level, muscle aches, pain, stiffness or weakness, heavier or irregular menstrual periods, thinning hair, slowed heart rate, depression, impaired memory.",
        "Autoimmune disease, thyroid surgery, radiation therapy, certain medications, pregnancy, congenital disease, pituitary disorder, iodine deficiency, age, sex (women are more likely to develop hypothyroidism)."
    )
    
    with st.form("thyroid_form"):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("Enter Patient Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = display_input('Age', 'Age in years', 'age', 1.0, 100.0, 'slider')
            sex = display_input('Gender', 'Gender of the person', 'sex', type='select', options=['Female', 'Male'])
            sex = 1 if sex == 'Male' else 0
            on_thyroxine = display_input('On Thyroxine', 'Is the person on thyroxine medication?', 'on_thyroxine', type='select', options=['No', 'Yes'])
            on_thyroxine = 1 if on_thyroxine == 'Yes' else 0
        
        with col2:
            tsh = display_input('TSH Level', 'Thyroid Stimulating Hormone level', 'tsh', 0.005, 500.0, 'number')
            t3_measured = display_input('T3 Measured', 'Was T3 hormone measured?', 't3_measured', type='select', options=['No', 'Yes'])
            t3_measured = 1 if t3_measured == 'Yes' else 0
            t3 = display_input('T3 Level', 'Triiodothyronine hormone level', 't3', 0.0, 10.0, 'number')
            tt4 = display_input('TT4 Level', 'Total Thyroxine level', 'tt4', 0.0, 500.0, 'number')
        
        # Sample data button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            positive_sample = st.form_submit_button("Load & Predict with Positive Sample")
        with col2:
            negative_sample = st.form_submit_button("Load & Predict with Negative Sample")
        with col3:
            submitted = st.form_submit_button("Predict with Current Data")
        st.markdown('</div>', unsafe_allow_html=True)
        
    if submitted:
        try:
            thyroid_prediction = models['thyroid'].predict([[age, sex, on_thyroxine, tsh, t3_measured, t3, tt4]])
            display_result(
                thyroid_prediction[0], 
                "POSITIVE: The analysis indicates Hypo-Thyroid disease. Please consult with an endocrinologist.",
                "NEGATIVE: The analysis does not indicate Hypo-Thyroid disease."
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")