from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time

st.set_page_config(page_title="Chatbot",page_icon="book")
# col1, col2, col3 = st.columns([1,8,1])
# with col2:
st.image("https://maketheroadny.org/wp-content/uploads/2020/08/KYR-graphic.jpg")
with st.sidebar:
    st.title('Know Your Rights Framework')
    st.markdown("## About")
    st.markdown(
            "📖Our Know Your Right Framwork "
            """empowers Indian citizens
            to understand their legal rights and responsibilities, 
            fostering a more just and equitable society. """
        )
    
    
    # st.subheader('Models and parameters')
    # selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    # if selected_model == 'Llama2-7B':
    #     llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    # elif selected_model == 'Llama2-13B':
    #     llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    # temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    # top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    # max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
# st.markdown(
#     """
#     <style>
# div.stButton > button:first-child {
#     background-color: #ffd0d0;
# }
# div.stButton > button:active {
#     background-color: #ff6262;
# }
#    div[data-testid="stStatusWidget"] div button {
#         display: none;
#         }   
#     .reportview-container {
#             margin-top: -2em;
#         }
#         #MainMenu {visibility: hidden;}
#         .stDeployButton {display:none;}
#         footer {visibility: hidden;}
#         #stDecoration {display:none;}
#     button[title="View fullscreen"]{
#     visibility: hidden;}
#         </style>
# """,
#     unsafe_allow_html=True,
# )

def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True) 

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code":True,})
db = FAISS.load_local("ipc_vector_db", embeddings,allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 4})

prompt_template = """<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code and comman mans basic rights queries, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. if your explaining a process or steps it should be in 10 bullients You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be short if small question and respnoses should be big if content or user query is big, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead ask for more contect in repsonse. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver simple, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

# You can also use other LLMs options from https://python.langchain.com/docs/integrations/llms. Here I have used TogetherAI API
TOGETHER_AI_API= os.environ['TOGETHER_AI'] = "b16198f5d992f8de5975180eb36130d88716c589d4574d4a787932e8c00f3f42"
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

input_prompt = st.chat_input("Ask Your Question about Your Rights and Laws")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role":"user","content":input_prompt})

    with st.chat_message("assistant"):
        with st.status("Analyzing 💡...",expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "Content Is From PDF it may consist Inconsitency** \n\n\n"
        for chunk in result["answer"]:
            full_response+=chunk
            time.sleep(0.0001)
            
            message_placeholder.markdown(full_response+" ▌")
        # st.button('Reset Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role":"assistant","content":result["answer"]})

# import streamlit as st


# st.title("💬 Chatbot")
# st.caption("🚀 A Streamlit chatbot powered by OpenAI")
# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# if prompt := st.chat_input():
#     if not openai_api_key:
#         st.info("Please add your OpenAI API key to continue.")
#         st.stop()

#     client = OpenAI(api_key=openai_api_key)
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)
#     response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
#     msg = response.choices[0].message.content
#     st.session_state.messages.append({"role": "assistant", "content": msg})
#     st.chat_message("assistant").write(msg)