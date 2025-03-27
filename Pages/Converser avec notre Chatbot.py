import streamlit as st
import random
import time
import google.generativeai as genai


genai.configure(api_key="AIzaSyCM68WtQU0ZxAAbpHq3K8KrXlgBJ3bN6F0")

model = genai.GenerativeModel('gemini-1.5-flash')

system_prompt = """" \
"Tu es un spécialiste de santé." \
"Tu donnes des réponses précises en les replaçant" \
"dans le contexte actuel." \
"""
chat = model.start_chat(history=[{'role': 'user', 'parts': [system_prompt]}])


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Bonjour 👋! Quelle est ta question ?",
            "Salut ! As-tu une question à me poser ? 🤔",
            "Bonjour. En quoi puis je t'être utile ?  🤔",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("Je suis votre Chatbot de Santé")

st.chat_message("MNTbot", avatar=":material/psychology_alt:")
st.write("Bienvenue. Ce Chat vous aidera à mieux comprendre votre maladie.")

col1, col2 = st.columns(2)
with col1:
    st.write("Posez vos questions et notre Chatbot de Santé vous répondra.")
    st.write_stream(response_generator())

with col2:
    if st.button("Terminer la session", icon="❌"):
        st.write("Session terminée. Merci d'avoir utilisé MNT chatbot.")
        st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
<style>
    .st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Comment puis-je t'aider aujourd'hui ?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = chat.send_message(user_input)
        st.markdown(response.text)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "MNTbot", "content": response.text})

page = st.sidebar.success('Sélectionnez votre choix ci-dessus ⤴️')