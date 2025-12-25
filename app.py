import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="RAG-Based Personal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Elite" feel
st.markdown("""
    <style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono:wght@300;400&display=swap');
    
    .stApp {
        background-color: #050510;
        background-image: radial-gradient(#1a1a40 1px, transparent 1px);
        background-size: 40px 40px;
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        background: -webkit-linear-gradient(45deg, #00f260, #0575E6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 10px rgba(5, 117, 230, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0a1a;
        border-right: 1px solid #1f1f3a;
        box-shadow: 5px 0 15px rgba(0,0,0,0.5);
    }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #111122;
        color: #00ff9d;
        border: 1px solid #333366;
        border-radius: 5px;
        font-family: 'Roboto Mono', monospace;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00ff9d;
        box-shadow: 0 0 10px rgba(0, 255, 157, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00f260, #0575E6);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(5, 117, 230, 0.6);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
        margin-bottom: 10px;
    }
    div[data-testid="stChatMessageContent"] p {
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #050510; 
    }
    ::-webkit-scrollbar-thumb {
        background: #333366; 
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #0575E6; 
    }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "process_complete" not in st.session_state:
        st.session_state.process_complete = False

def get_files_text(uploaded_files):
    text = ""
    documents = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)

        if loader:
            documents.extend(loader.load())
            os.remove(temp_file_path)
    return documents

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vectorstore(text_chunks, model_provider, openai_api_key=None):
    if model_provider == "OpenAI":
        if not openai_api_key:
            st.error("OpenAI API Key is missing!")
            return None
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    else:  # Ollama
        embeddings = OllamaEmbeddings(model="llama3") # Default to llama3 or user choice

    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def get_conversation_chain(vectorstore, model_provider, openai_api_key=None):
    if model_provider == "OpenAI":
        llm = ChatOpenAI(temperature=0.5, openai_api_key=openai_api_key, model_name='gpt-3.5-turbo')
    else:
        llm = ChatOllama(model="llama3", temperature=0.5)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Lütfen önce döküman yükleyin ve işleyin.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user", avatar="👤"):
                st.write(message.content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message.content)

def main():
    init_session_state()

    with st.sidebar:
        st.title("🎛️ Kontrol Paneli")
        st.markdown("---")
        
        # Model Selection
        model_provider = st.selectbox(
            "Model Sağlayıcı Seçin",
            ("OpenAI", "Ollama (Yerel)")
        )

        openai_api_key = ""
        if model_provider == "OpenAI":
            openai_api_key = st.text_input("OpenAI API Key", type="password", help="API anahtarınız güvenli bir şekilde kullanılacaktır.")
            if not openai_api_key:
                st.info("Devam etmek için API Key giriniz.")

        st.markdown("### 📄 Döküman Yükleme")
        uploaded_files = st.file_uploader(
            "PDF veya TXT dosyalarınızı buraya bırakın",
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        
        process_button = st.button("Analiz Et ve İşle")

        if process_button and uploaded_files:
            if model_provider == "OpenAI" and not openai_api_key:
                st.error("Lütfen OpenAI API Key giriniz.")
            else:
                with st.spinner("Dökümanlar işleniyor..."):
                    # 1. Get raw text
                    raw_documents = get_files_text(uploaded_files)
                    
                    # 2. Get text chunks
                    text_chunks = get_text_chunks(raw_documents)
                    
                    # 3. Create vector store
                    vectorstore = get_vectorstore(text_chunks, model_provider, openai_api_key)
                    
                    if vectorstore:
                        # 4. Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore, model_provider, openai_api_key)
                        st.session_state.process_complete = True
                        st.success("İşlem Tamamlandı! Artık sorularınızı sorabilirsiniz.")

        st.markdown("---")
        st.markdown("### 👨‍💻 Geliştirici")
        st.info("**Bahattin Yunus Çetin**\n\nIT Architect")

    st.title("🤖 RAG-Based Personal AI Assistant")
    st.markdown("Dökümanlarınızla güvenli ve akıllıca sohbet edin.")

    # Chat interface
    user_question = st.chat_input("Dökümanlarınız hakkında bir soru sorun...")
    
    if user_question:
        if not st.session_state.process_complete:
            st.warning("Sohbete başlamadan önce lütfen döküman yükleyin ve 'Analiz Et' butonuna tıklayın.")
        else:
            handle_userinput(user_question)

    # Display clean slate if no messages
    if not st.session_state.messages and not user_question and not st.session_state.process_complete:
        st.markdown("""
        #### Nasıl Kullanılır?
        1. Sol menüden **Model** seçin.
        2. **Dökümanlarınızı** (PDF/TXT) yükleyin.
        3. **'Analiz Et ve İşle'** butonuna basın.
        4. Sohbet kutusundan sorularınızı sorun!
        """)

if __name__ == '__main__':
    main()
