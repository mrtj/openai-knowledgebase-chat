import time
import streamlit as st
import streamlit_chat

from chatbot import ChatBot, KnowledgeBase

TEXTS = {
    'it': {
        'Message:': 'Messaggio:',
        '{chatbot_name} Virtual Assistant': 'Assistente Virtuale {chatbot_name}',
        'Please enter your OpenAI API key:': 'Inserisci la tua OpenAI API key:',
        '{chatbot_name} is responding ...': '{chatbot_name} sta scrivendo ...'
    },
    'en': {
        'Message:': 'Message:',
        '{chatbot_name} Virtual Assistant': '{chatbot_name} Virtual Assistant',
        'Please enter your OpenAI API key:': 'Please enter your OpenAI API key:',
        '{chatbot_name} is responding ...': '{chatbot_name} is responding ...',
    }
}

def main():

    with st.sidebar:
        chatbot_name = st.selectbox('ChatBot:', ChatBot.knowledge_bases())

    kb = KnowledgeBase(kb_name=chatbot_name)
    lang = kb.language if kb.language in TEXTS else 'en'
    texts = TEXTS[lang]

    st.title(texts['{chatbot_name} Virtual Assistant'].format(chatbot_name=chatbot_name))

    if 'agent' not in st.session_state:
        if 'openai_api_key' not in st.session_state or len(st.session_state.openai_api_key) == 0:
            openai_api_key = st.text_input(
                texts['Please enter your OpenAI API key:'], type='password', key='openai_api_key'
            )
            return
        else:
            openai_api_key = st.session_state.openai_api_key
            st.session_state.agent = ChatBot(knowledge_base=kb, openai_api_key=openai_api_key)

    agent: ChatBot = st.session_state.agent

    msg_idx = 0
    def add_message(text, is_user):
        nonlocal msg_idx
        seed = '47' if is_user else None
        avatar_style = 'big-ears-neutral' if is_user else None
        streamlit_chat.message(
            text, key='msg' + str(msg_idx), is_user=is_user,
            seed=seed, avatar_style=avatar_style
        )
        msg_idx += 1

    st.write(kb.help_text)

    for item in agent.history:
        if item['role'] == 'assistant':
            is_user = False
        elif item['role'] == 'user':
            is_user = True
        else:
            continue
        add_message(item['content'], is_user=is_user)

    if 'question' in st.session_state and st.session_state.question:
        question = st.session_state.question
        print(f'USER:\n{question}\n')
        add_message(question, is_user=True)
        st.session_state.question = ''
        with st.spinner(texts['{chatbot_name} is responding ...'].format(chatbot_name=chatbot_name)):
            answer = agent.answer_question(question, debug=True)
        print(f'ASSISTANT:\n{answer}\n')
        add_message(answer, is_user=False)

    st.text_input(texts['Message:'], key='question')

if __name__ == '__main__':
    main()
