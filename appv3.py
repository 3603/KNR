import os
import streamlit as st
import torch
import speech_recognition as sr
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import time

# Set up Whisper speech-to-text pipeline
st.cache_resource
def load_speech_to_text_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-large", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

speech_to_text_model = load_speech_to_text_model()

st.set_page_config(page_title="KWR", page_icon="book")

# Sidebar for navigation
st.sidebar.title("Navigation Panel")
menu = st.sidebar.radio("Select an option", ("Home", "About", "Contact Us"))

if menu == "Home":
    st.image("https://maketheroadny.org/wp-content/uploads/2020/08/KYR-graphic.jpg")

    # Main chatbot interface
    def reset_conversation():
        st.session_state.messages = []
        st.session_state.memory.clear()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True})
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt_template = """<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code and common man's basic rights queries, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. If you're explaining a process or steps, it should be in 10 bullet points. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be short if the question is small and responses should be big if the content or user query is big, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead ask for more context in response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver simple, precise, and contextually relevant information pertaining to the Indian Penal Code.
    CONTEXT: {context}
    CHAT HISTORY: {chat_history}
    QUESTION: {question}
    ANSWER:
    </s>[INST]
    """

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['context', 'question', 'chat_history'])

    TOGETHER_AI_API = os.environ['TOGETHER_AI'] = "b16198f5d992f8de5975180eb36130d88716c589d4574d4a787932e8c00f3f42"
    llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5,
        max_tokens=1024,
        together_api_key=f"{TOGETHER_AI_API}"
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    for message in st.session_state.messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))

    # Speech Recognition using microphone
    def recognize_speech():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source)

        try:
            # Recognize speech using Whisper model
            st.info("Transcribing speech...")
            audio_data = audio.get_wav_data()
            transcription = speech_to_text_model(audio_data)["text"]
            st.success(f"Transcription: {transcription}")
            return transcription
        except Exception as e:
            st.error(f"Error during transcription: {e}")
            return ""

    # Add a button for speech input
    if st.button("🎤 Click to Speak"):
        input_prompt = recognize_speech()
        if input_prompt:
            with st.chat_message("user"):
                st.write(input_prompt)

            st.session_state.messages.append({"role": "user", "content": input_prompt})

            with st.chat_message("assistant"):
                with st.status("Analyzing 💡...", expanded=True):
                    result = qa.invoke(input=input_prompt)

                    message_placeholder = st.empty()

                    full_response = "Content Is From PDF; it may consist of inconsistencies** \n\n\n"
                    for chunk in result["answer"]:
                        full_response += chunk
                        time.sleep(0.0001)

                        message_placeholder.markdown(full_response + " ▌")
                    st.button('Reset Chat 🗑️', on_click=reset_conversation)

            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

elif menu == "About":
    st.header("About This Application")
    st.write("""
        This application serves as a chatbot for users seeking information about their rights under the Indian Penal Code. 
        The chatbot utilizes advanced language models to provide accurate and concise responses based on user queries.
    """)

elif menu == "Contact Us":
    st.header("Contact Us")
    st.write("""
        For any inquiries or feedback, please reach out via email at example@example.com.
    """)

