import streamlit as st
import pickle
import pandas as pd

# Titre de l'application
st.title("Prédiction Mental Health avec HistGradientBoosting et XGBoost")

# Charger les modèles sauvegardés
@st.cache(allow_output_mutation=True)
def load_models():
    with open('saved_models/hist_model.pkl', 'rb') as f:
        hist_model = pickle.load(f)
    with open('saved_models/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    return hist_model, xgb_model

hist_model, xgb_model = load_models()

# Formulaire pour saisir les features
st.sidebar.header("Saisir les valeurs des features")

# Saisie des features
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
country = st.sidebar.selectbox("Country", ["USA", "Canada", "UK", "Australia"])
occupation = st.sidebar.selectbox("Occupation", ["Engineer", "Doctor", "Teacher", "Student"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
family_history = st.sidebar.selectbox("Family History", ["Yes", "No"])
treatment = st.sidebar.selectbox("Treatment", ["Yes", "No"])
days_indoors = st.sidebar.number_input("Days Indoors", min_value=0, max_value=365, value=0)
growing_stress = st.sidebar.selectbox("Growing Stress", ["Yes", "No"])
changes_habits = st.sidebar.selectbox("Changes Habits", ["Yes", "No"])
mental_health_history = st.sidebar.selectbox("Mental Health History", ["Yes", "No"])
mood_swings = st.sidebar.selectbox("Mood Swings", ["Yes", "No"])
coping_struggles = st.sidebar.selectbox("Coping Struggles", ["Yes", "No"])
work_interest = st.sidebar.selectbox("Work Interest", ["Yes", "No"])
social_weakness = st.sidebar.selectbox("Social Weakness", ["Yes", "No"])
mental_health_interview = st.sidebar.selectbox("Mental Health Interview", ["Yes", "No"])
care_options = st.sidebar.selectbox("Care Options", ["Yes", "No"])

# Bouton pour faire la prédiction
if st.sidebar.button("Prédire"):
    # Créer un DataFrame avec les valeurs saisies
    input_data = pd.DataFrame([[
        gender, country, occupation, self_employed, family_history, treatment,
        days_indoors, growing_stress, changes_habits, mental_health_history,
        mood_swings, coping_struggles, work_interest, social_weakness,
        mental_health_interview, care_options
    ]], columns=[
        'Gender', 'Country', 'Occupation', 'self_employed', 'family_history',
        'treatment', 'Days_Indoors', 'Growing_Stress', 'Changes_Habits',
        'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles',
        'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options'
    ])

    # Encoder les variables catégorielles
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for col in input_data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        input_data[col] = le.fit_transform(input_data[col])
        label_encoders[col] = le

    # Faire la prédiction avec HistGradientBoostingClassifier
    hist_pred = hist_model.predict(input_data)
    st.write("Prédiction (HistGradientBoosting) :", hist_pred[0])

    # Faire la prédiction avec XGBoost
    xgb_pred = xgb_model.predict(input_data)
    st.write("Prédiction (XGBoost) :", xgb_pred[0])