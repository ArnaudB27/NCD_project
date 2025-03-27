import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import io
import xlsxwriter

st.title('Prédiction des risques de Mortalité du patient.')

page = st.sidebar.success('Sélectionnez votre choix ci-dessus ⤴️')

file_path = 'dataset.csv' 
df = pd.read_csv(file_path)

if 'Unnamed: 83' in df.columns:
    df.drop(["Unnamed: 83"], axis=1, inplace=True)

exclude_columns = ['hospital_id', 'patient_id', 'icu_id', 'encounter_id']
df = df.drop(columns=exclude_columns)
df = df.dropna()

columns_to_convert = [
    'age', 'bmi', 'elective_surgery', 'intubated_apache', 'arf_apache', 'gcs_unable_apache',
    'ventilated_apache', 'd1_diasbp_max', 'd1_diasbp_min', 'd1_diasbp_noninvasive_min', 
    'd1_diasbp_noninvasive_max', 'd1_heartrate_max', 'd1_heartrate_min', 'd1_mbp_max', 
    'd1_mbp_min', 'd1_mbp_noninvasive_min', 'd1_mbp_noninvasive_max', 'd1_resprate_max', 
    'd1_resprate_min', 'lymphoma', 'solid_tumor_with_metastasis', 'leukemia', 'immunosuppression', 
    'hepatic_failure', 'diabetes_mellitus', 'cirrhosis', 'aids', 'd1_glucose_min', 'd1_glucose_max', 
    'h1_sysbp_noninvasive_min', 'h1_sysbp_noninvasive_max', 'h1_sysbp_min', 'h1_sysbp_max', 
    'h1_spo2_min', 'h1_spo2_max', 'apache_2_diagnosis', 'gcs_eyes_apache', 'gcs_motor_apache', 
    'heart_rate_apache', 'gcs_verbal_apache', 'map_apache', 'd1_spo2_max', 'd1_spo2_min', 
    'd1_sysbp_max', 'd1_sysbp_min', 'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min', 
    'h1_diasbp_max', 'h1_diasbp_min', 'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min', 
    'h1_heartrate_max', 'h1_heartrate_min', 'h1_mbp_max', 'h1_mbp_min', 'h1_mbp_noninvasive_max', 
    'h1_mbp_noninvasive_min', 'h1_resprate_max', 'h1_resprate_min'
]

df[columns_to_convert] = df[columns_to_convert].astype(int)

column_rename = {
    'age': 'Âge',
    'bmi': 'Indice de Masse Corporelle (IMC)',
    'elective_surgery': 'Chirurgie élective',
    'height': 'Taille (en cm)',
    'pre_icu_los_days': 'Durée avant admission en soins intensifs (en jours)',
    'weight': 'Poids (en kg)',
    'apache_2_diagnosis': 'Diagnostic Apache 2',
    'apache_3j_diagnosis': 'Diagnostic Apache 3J',
    'apache_post_operative': 'Diagnostic postopératoire Apache',
    'arf_apache': 'Insuffisance rénale aiguë Apache',
    'gcs_eyes_apache': 'GCS - Réponse oculaire',
    'gcs_motor_apache': 'GCS - Réponse motrice',
    'gcs_unable_apache': 'GCS - Réponse impossible',
    'gcs_verbal_apache': 'GCS - Réponse verbale',
    'heart_rate_apache': 'Fréquence cardiaque (Apache)',
    'intubated_apache': 'Intubation (Apache)',
    'map_apache': 'Pression artérielle moyenne (Apache)',
    'resprate_apache': 'Fréquence respiratoire (Apache)',
    'temp_apache': 'Température corporelle (Apache)',
    'ventilated_apache': 'Ventilation mécanique (Apache)',
    'd1_diasbp_max': 'Pression artérielle diastolique maximale (Jour 1)',
    'd1_diasbp_min': 'Pression artérielle diastolique minimale (Jour 1)',
    'd1_diasbp_noninvasive_max': 'Pression artérielle diastolique non invasive maximale (Jour 1)',
    'd1_diasbp_noninvasive_min': 'Pression artérielle diastolique non invasive minimale (Jour 1)',
    'd1_heartrate_max': 'Fréquence cardiaque maximale (Jour 1)',
    'd1_heartrate_min': 'Fréquence cardiaque minimale (Jour 1)',
    'd1_mbp_max': 'Pression artérielle moyenne maximale (Jour 1)',
    'd1_mbp_min': 'Pression artérielle moyenne minimale (Jour 1)',
    'd1_mbp_noninvasive_max': 'Pression artérielle moyenne non invasive maximale (Jour 1)',
    'd1_mbp_noninvasive_min': 'Pression artérielle moyenne non invasive minimale (Jour 1)',
    'd1_resprate_max': 'Fréquence respiratoire maximale (Jour 1)',
    'd1_resprate_min': 'Fréquence respiratoire minimale (Jour 1)',
    'd1_spo2_max': 'Saturation en oxygène maximale (Jour 1)',
    'd1_spo2_min': 'Saturation en oxygène minimale (Jour 1)',
    'd1_sysbp_max': 'Pression artérielle systolique maximale (Jour 1)',
    'd1_sysbp_min': 'Pression artérielle systolique minimale (Jour 1)',
    'd1_sysbp_noninvasive_max': 'Pression artérielle systolique non invasive maximale (Jour 1)',
    'd1_sysbp_noninvasive_min': 'Pression artérielle systolique non invasive minimale (Jour 1)',
    'd1_temp_max': 'Température corporelle maximale (Jour 1)',
    'd1_temp_min': 'Température corporelle minimale (Jour 1)',
    'h1_diasbp_max': 'Pression artérielle diastolique maximale (Jour 1 - Heure 1)',
    'h1_diasbp_min': 'Pression artérielle diastolique minimale (Jour 1 - Heure 1)',
    'h1_diasbp_noninvasive_max': 'Pression artérielle diastolique non invasive maximale (Jour 1 - Heure 1)',
    'h1_diasbp_noninvasive_min': 'Pression artérielle diastolique non invasive minimale (Jour 1 - Heure 1)',
    'h1_heartrate_max': 'Fréquence cardiaque maximale (Jour 1 - Heure 1)',
    'h1_heartrate_min': 'Fréquence cardiaque minimale (Jour 1 - Heure 1)',
    'h1_mbp_max': 'Pression artérielle moyenne maximale (Jour 1 - Heure 1)',
    'h1_mbp_min': 'Pression artérielle moyenne minimale (Jour 1 - Heure 1)',
    'h1_mbp_noninvasive_max': 'Pression artérielle moyenne non invasive maximale (Jour 1 - Heure 1)',
    'h1_mbp_noninvasive_min': 'Pression artérielle moyenne non invasive minimale (Jour 1 - Heure 1)',
    'h1_resprate_max': 'Fréquence respiratoire maximale (Jour 1 - Heure 1)',
    'h1_resprate_min': 'Fréquence respiratoire minimale (Jour 1 - Heure 1)',
    'h1_spo2_max': 'Saturation en oxygène maximale (Jour 1 - Heure 1)',
    'h1_spo2_min': 'Saturation en oxygène minimale (Jour 1 - Heure 1)',
    'h1_sysbp_max': 'Pression artérielle systolique maximale (Jour 1 - Heure 1)',
    'h1_sysbp_min': 'Pression artérielle systolique minimale (Jour 1 - Heure 1)',
    'h1_sysbp_noninvasive_max': 'Pression artérielle systolique non invasive maximale (Jour 1 - Heure 1)',
    'h1_sysbp_noninvasive_min': 'Pression artérielle systolique non invasive minimale (Jour 1 - Heure 1)',
    'd1_glucose_max': 'Glycémie maximale (Jour 1)',
    'd1_glucose_min': 'Glycémie minimale (Jour 1)',
    'd1_potassium_max': 'Taux de potassium maximal (Jour 1)',
    'd1_potassium_min': 'Taux de potassium minimal (Jour 1)',
    'apache_4a_hospital_death_prob': 'Probabilité de décès hospitalier (Apache 4A)',
    'apache_4a_icu_death_prob': 'Probabilité de décès en réanimation (Apache 4A)',
    'aids': 'VIH/SIDA',
    'cirrhosis': 'Cirrhose',
    'diabetes_mellitus': 'Diabète sucré',
    'hepatic_failure': 'Insuffisance hépatique',
    'immunosuppression': 'Immunosuppression',
    'leukemia': 'Leucémie',
    'lymphoma': 'Lymphome',
    'solid_tumor_with_metastasis': 'Tumeur solide avec métastases',
    'ethnicity': 'Origine ethnique',
    'gender': 'Genre',
    'icu_admit_source': 'Source d\'admission en réanimation',
    'icu_stay_type': 'Type de séjour en réanimation',
    'icu_type': 'Type de soins en réanimation',
    'apache_3j_bodysystem': 'Système corporel Apache 3J',
    'apache_2_bodysystem': 'Système corporel Apache 2'
}

df.rename(columns=column_rename, inplace=True)
df.columns.tolist()

duplicate_rows = df[df.duplicated()]
if not duplicate_rows.empty:
    df = df.drop_duplicates()

target = 'hospital_death'
numeric_features = df.select_dtypes(include=['int', 'float']).columns.tolist()
numeric_features.remove(target) 
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

X = df.drop([target], axis=1)
y = df[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)


@st.cache_data
def train_model():
    model_pipeline.fit(X_train, y_train)
    return model_pipeline


model_pipeline = train_model()

y_pred = model_pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

st.write(f'Précision du modèle : {accuracy * 100:.2f}%')

st.header('Faire une prédiction')

input_data = {}


def is_binary_column(column):
    return set(df[column].unique()) == {0, 1}


for feature in numeric_features:
    if is_binary_column(feature):
        input_data[feature] = st.selectbox(feature, options=[0, 1])
    else:
        default_value = df[feature].mean() if df[feature].dtype != 'object' else 0
        input_data[feature] = st.number_input(feature, value=default_value)

for feature in categorical_features:
    input_data[feature] = st.selectbox(feature, options=['missing'] + df[feature].dropna().unique().tolist())


@st.cache_data # IMPORTANT: Cache the conversion to prevent computation on every rerun
def convert_for_download(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
    processed_data = output.getvalue()
    return processed_data


dfDownload = pd.DataFrame.from_dict(input_data, orient='index')

st.sidebar.download_button(
    label="Download Excel file",
    data=convert_for_download(dfDownload),
    file_name="data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    icon=":material/download:",
)


if st.button('Faire une prédiction'):
    input_df = pd.DataFrame([input_data])
    prediction = model_pipeline.predict(input_df)
    with st.spinner("Analyse en cours...", show_time=True):
        time.sleep(5)
    st.sidebar.info(f"Prédiction de la mortalité : {prediction[0]}", icon="🚨")
