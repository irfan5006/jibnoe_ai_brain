# 집뇌 JIBNOE — Seoul Housing Brain AI

집뇌 JIBNOE — Seoul Housing Brain AI is an AI-powered housing price prediction system for apartments in Seoul.

The goal of this project is to estimate apartment prices using important features such as area (m²) and market segment. The system is designed to support real estate professionals, investors, and buyers in making data-driven decisions.

---

## Project Overview

The project predicts apartment prices across six districts in Seoul using historical housing data. Machine learning models are applied to identify key factors influencing property prices and generate reliable predictions.

---

## Dataset

Source: Real estate listings in Seoul

The dataset consists of six CSV files, each representing a different district.

Columns:

- District – Apartment location  
- Area_m2 – Size in square meters  
- Market_Segment – Apartment category (Luxury / Standard / Economy)  
- Price – Target variable (apartment price)

---

## Technologies and Tools

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

![Seaborn](https://img.shields.io/badge/Seaborn-DataVisualization-blue?style=for-the-badge)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## Model Performance

Final Model: Random Forest Regressor

After performing hyperparameter tuning, the model achieved:

Training Accuracy: 90%  
Testing Accuracy: 88%

Random Forest was selected as the final model because of its ability to capture non-linear relationships in housing price data and its strong predictive performance.

---

## Project Workflow

1. Data Collection and Integration  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering  
4. Train/Test Split  
5. Model Training  
6. Model Evaluation using RMSE and R²  
7. Ensemble Learning  
8. Hyperparameter Tuning  
9. Model Saving using Joblib  
10. Optional Deployment using Streamlit

---

## Key Insights

- Apartment area (m²) is one of the strongest predictors of price  
- Market segment significantly affects property value  
- Property prices vary across districts in Seoul  
- Ensemble models improve prediction performance

---

## Future Improvements

- Add additional features such as number of rooms, year built, and amenities  
- Experiment with advanced models such as XGBoost or LightGBM  
- Deploy a full web application for real-time predictions

---

## Author

Muhammad Irfan

LinkedIn:  
https://www.linkedin.com/in/m-erfaan

---

## License

MIT License
