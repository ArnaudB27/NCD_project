import streamlit as st
import plotly.express as px
import pandas as pd

# Charger les données
df0 = pd.read_csv('TableCauseMortalité2.csv', sep=';', encoding='latin1', encoding_errors='replace')
df0['Cum_Sum'] = df0['OBS_VALUE'].cumsum()
df = df0[(df0['DEATH_CAUSE'] != '_T') & (df0['Calculation methodology'] == 'Taux standardisé') & (df0['Sex.1'] != 'Total')]
df1 = df0[(df0['DEATH_CAUSE'] != '_T') & (df0['Calculation methodology'] == 'Taux standardisé') & (df0['Group'] == 'Maladies Non Transmissibles') & (df0['Sex.1'] != 'Total')]

# Ajouter une boîte de sélection pour choisir le type de graphique
country = st.selectbox('Choisissez un pays', ['KOR', 'GRC', 'NLD', 'RUS', 'ITA', 'EST', 'USA', 'AUT', 'ISR', 'NZL', 'PRT', 'ARG', 'POL', 'ZAF', 'DNK', 'ISL', 'CHL', 'ESP', 'GBR', 'CHE', 'CAN', 'BRA', 'SVN', 'CZE', 'LUX', 'IRL', 'MEX', 'FRA', 'PER', 'DEU', 'SWE', 'CRI', 'AUS', 'JPN', 'HUN', 'BGR', 'ROU', 'HRV', 'LVA', 'LTU', 'NOR', 'COL', 'BEL', 'FIN', 'TUR', 'SVK'])
chart_type = st.selectbox('Choisissez votre graphique', ['Causes Mortalité', 'Classement MNT'])

# Créer le graphique
if chart_type == 'Causes Mortalité':
    fig = px.bar(df[df['REF_AREA'] == country], x='Cause of Death', y='Cum_Sum', color='Sex.1', barmode='group', orientation='h')
elif chart_type == 'Classement MNT':
    fig = px.bar(df1[df1['REF_AREA'] == country], x='Cause of Death', y='Cum_Sum', color='Sex.1', barmode='group', orientation='v')

# Afficher le graphique
st.plotly_chart(fig, use_container_width=True)

page = st.sidebar.success('Sélectionnez votre choix ci-dessus ⤴️')