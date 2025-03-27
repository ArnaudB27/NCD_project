import streamlit as st
st.set_page_config(page_title="MNT App", page_icon='⚕️', layout="wide",)
st.title("👩‍⚕️Bienvenu sur l'application MNT👨‍⚕️")
st.header("Maladies non transmissibles: défi majeur pour les hôpitaux")
st.markdown("Développée par Valentine, Sandra et Arnaud")
st.write("Aujourd’hui, les maladies non transmissibles (MNT) représentent une part considérable des consultations médicales et des hospitalisations.")
st.write("En Europe, elles sont responsables de près de 90/%/ des décès chaque année, avec des pathologies comme les maladies cardiovasculaires, les cancers, le diabète et les maladies respiratoires chroniques en première ligne.")
st.write("Selon l’Organisation mondiale de la santé (OMS), plus de 77/%/ des consultations médicales concernent ces maladies, mettant une pression croissante sur les professionnels de santé et les établissements hospitaliers.")
st.image("MNT.jpg")
page = st.sidebar.success('Sélectionnez votre choix ci-dessus ⤴️')