import streamlit as st
import streamlit_chat

from chatbot import ChatBot

st.title('Virtual Assistant')

if not 'agent' in st.session_state:
    openai_api_key = st.secrets["api_secret"]
    st.session_state.agent = ChatBot(kb_name='empatair', openai_api_key=openai_api_key)

agent: ChatBot = st.session_state.agent

def submit():
    question = st.session_state.question
    agent.answer_question(question)
    st.session_state.question = ''

for i, item in enumerate(agent.history):
    if item['role'] == 'assistant':
        is_user = False
    elif item['role'] == 'user':
        is_user = True
    else:
        continue
    streamlit_chat.message(item['content'], key='msg' + str(i), is_user=is_user)

st.text_input('You: ', key='question', on_change=submit)
