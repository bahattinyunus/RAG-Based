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
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton > button:hover {
        background-color: #45a049;
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
