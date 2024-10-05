import streamlit as st
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

st.set_page_config(page_title="Know Your Rights", page_icon="⚖️", layout="wide")
# Load environment variables
load_dotenv()
TOGETHER_AI_API = os.getenv("TOGETHER_AI_API")
os.getenv("TOGETHER_AI_API") == "b16198f5d992f8de5975180eb36130d88716c589d4574d4a787932e8c00f3f42"
# Set up Streamlit page config with navigation


# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Legal ChatBot", "About"])

# Custom CSS for better UI/UX
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
        border-radius: 10px;
    }
    div.stButton > button:hover {
        background-color: #0056b3;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .reportview-container {
        margin-top: -2em;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#007BFF, #0056b3);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Introduction Page
if page == "Introduction":
    st.title("Welcome to Know Your Rights ChatBot")
    st.write("""
        This platform helps you understand your legal rights with a focus on the Indian Penal Code (IPC) 
        and common man's basic rights. Ask any question, and our AI will assist you with relevant, concise 
        legal information.
    """)
    st.image("https://maketheroadny.org/wp-content/uploads/2020/08/KYR-graphic.jpg", caption="Know Your Rights")

# Legal ChatBot Page
elif page == "Legal ChatBot":
    st.title("Legal ChatBot - Ask About Your Rights")
    
    # Reset conversation function
    def reset_conversation():
        st.session_state.messages = []
        st.session_state.memory.clear()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

    # Hugging Face Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True})
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Prompt Template
    prompt_template = """
    <s>[INST]This is a chat template and as a legal chatbot specializing in Indian Penal Code and common man's basic rights queries, your primary objective is to provide accurate and concise information based on the user's questions. 
    You will provide answers based on IPC, in 10 bullet points for processes or steps. Your responses should be concise for short queries, and detailed if the user's query demands it.
    CONTEXT: {context}
    CHAT HISTORY: {chat_history}
    QUESTION: {question}
    ANSWER:
    </s>[INST]
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

    # Together AI with Mistral Model
    TOGETHER_AI_API = os.getenv("TOGETHER_AI_API")
    llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5,
        max_tokens=1024,
        together_api_key=f"{TOGETHER_AI_API}"
    )

    # Conversational Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    # Display conversation
    for message in st.session_state.messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))

    # Input prompt from user
    input_prompt = st.chat_input("Ask Your Question about Your Rights and Laws")

    if input_prompt:
        with st.chat_message("user"):
            st.write(input_prompt)

        st.session_state.messages.append({"role": "user", "content": input_prompt})

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.status("Analyzing 💡...", expanded=True):
                result = qa.invoke(input=input_prompt)

                message_placeholder = st.empty()
                full_response = "Content Is From PDF it may consist of Inconsistencies**\n\n\n"
                
                for chunk in result["answer"]:
                    full_response += chunk
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " ▌")

        st.button('Reset All Chat 🗑️', on_click=reset_conversation)
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

# About Page
elif page == "About":
    st.title("About Know Your Rights")
    st.write("""
        The 'Know Your Rights' chatbot is designed to help users understand their legal rights under the Indian Penal Code.
        Using AI-powered models like Mistral and Hugging Face, it retrieves relevant legal context and provides concise, 
        accurate responses to user queries.
    """)
    st.write("Developed by [Your Name](https://your-portfolio.com)")
