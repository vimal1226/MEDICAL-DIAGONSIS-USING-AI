# Medical AI Diagnosis System

![Project Logo](https://cdn-icons-png.flaticon.com/512/4807/4807695.png)

A Streamlit-based web application that utilizes machine learning models to predict the likelihood of various medical conditions. Designed for educational and demonstration purposes.

## Features

- 🩺 **Five Disease Predictors**:
  - Diabetes Prediction
  - Heart Disease Risk Assessment
  - Parkinson's Disease Detection (voice analysis)
  - Lung Cancer Risk Evaluation
  - Hypo-Thyroid Diagnosis

- 💻 **Interactive Web Interface**:
  - Sidebar navigation
  - Dynamic input forms
  - Sample data loading
  - Real-time predictions
  - Detailed disease information

- 🎨 **Professional UI**:
  - Responsive design
  - Custom styling
  - Background overlay
  - Clear result visualization

## Technologies and Tools Used

- **Frontend**: Streamlit for UI development
- **Backend**: Python and Flask for API integration
- **Machine Learning**: Scikit-learn, XGBoost, and TensorFlow for model training
- **Data Processing**: Pandas, NumPy for data manipulation
- **Database**: MongoDB for storing user data (optional)
- **Version Control**: Git and GitHub for collaboration
- **Deployment**: Streamlit Cloud, Heroku, or AWS

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
```bash
git clone https://github.com/yourusername/medical-ai-diagnosis.git
cd medical-ai-diagnosis
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app.py
```

## Usage

1. Launch the application.
2. Select a disease from the sidebar.
3. Fill in patient information.
4. Click "Predict" for diagnosis.
5. Review results and recommendations.

> **Note:** This tool is for educational purposes only. Always consult a healthcare professional for medical advice.

## Disease Models

| Disease        | Algorithm Used | Dataset Source                          |
|---------------|---------------|-----------------------------------------|
| Diabetes      | SVM           | Pima Indians Diabetes Dataset          |
| Heart Disease| Random Forest | Cleveland Clinic Foundation            |
| Parkinson's  | XGBoost       | UCI ML Parkinsons Dataset              |
| Lung Cancer  | Logistic Reg  | Kaggle Survey Data                     |
| Hypo-Thyroid | Decision Tree | UCI Thyroid Disease Records            |

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- Dataset providers: UCI Machine Learning Repository, Kaggle
- Streamlit for the web framework
- Scikit-learn for machine learning tools
- Icons by Flaticon
- Medical research organizations advancing AI in healthcare



> **Disclaimer:** The predictions from this system should not be considered medical advice. Always consult qualified healthcare professionals for medical diagnoses and treatment.

