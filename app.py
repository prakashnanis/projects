import streamlit as st
from gtts import gTTS
from PyPDF2 import PdfReader
import docx
import pandas as pd
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from streamlit_lottie import st_lottie
from dotenv import load_dotenv
import pyttsx3
import speech_recognition as sr
import uuid

backend_url = "http://127.0.0.1:8501"  # Ensure this matches your backend URL

load_dotenv()


# Function to extract text from PDF files
@st.cache_data(show_spinner=False)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from Word files
@st.cache_data(show_spinner=False)
def get_docx_text(docx_files):
    text = ""
    for doc in docx_files:
        doc_reader = docx.Document(doc)
        for para in doc_reader.paragraphs:
            text += para.text + "\n"
    return text

# Function to extract text from Excel files
@st.cache_data(show_spinner=False)
def get_excel_text(excel_files):
    text = ""
    for excel in excel_files:
        df = pd.read_excel(excel)
        text += df.to_string(index=False)
    return text

# Function to split text into chunks
@st.cache_data(show_spinner=False)
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store using FAISS
@st.cache_data(show_spinner=False)
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input question and retrieve answer
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to convert text to speech using pyttsx3
def text_to_speech(text):
    if not st.session_state.get('mute', False):  # Check the mute state
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)  # Speed percent (can go over 100)
        engine.setProperty('volume', 1.0)  # Volume 0-1
        engine.say(text)
        engine.runAndWait()
        
# Function to load Lottie animation from a JSON file
def load_lottie_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Function to handle voice commands
def get_voice_command():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        st.info("Listening for your question...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        st.success(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        st.error("Sorry, I did not understand that.")
        return ""
    except sr.RequestError:
        st.error("Could not request results; check your network connection.")
        return ""

# Function to register user
def register_user(email, password):
    response = requests.post(f"{backend_url}/register", json={"email": email, "password": password})
    return response

# Function to login user
def login_user(email, password):
    response = requests.post(f"{backend_url}/login", json={"email": email, "password": password})
    return response

# Function to provide introduction
def introduce_dexter(animation):
    st_lottie(animation, height=300, key="dexter")
    # Use text-to-speech to read the introduction text
    introduction = "Hello, I am Dexter, your virtual assistant. I can help you process documents, answer questions based on the content, and more. Let's get started!"
    text_to_speech(introduction)
    # Set session state to show the introduction text after animation
    st.session_state['animation_done'] = True

# Main Streamlit application
def main():
    st.set_page_config(page_title="Document Chatbot", layout="wide")

    st.markdown(
        """
        <style>
        .main {
            background-color: #fff;
            padding: 2rem;
            font-family: Arial, sans-serif;
            color: #333;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2E7D32,#1B5E20);
            color: white;
        }
        .stButton>button {
            color: white;
            background-color: #2E7D32;
            border: None;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #1B5E56;
            border: None;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .header h1 {
            font-size: 3rem;
            color: #2E7D32;
        }
        .header p {
            font-size: 1.2rem;
            color: #666;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .question-input {
            width: 80%;
            margin-bottom: 1rem;
        }
        .response {
            width: 80%;
            padding: 1rem;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-top: 1rem;
        }
        .response h3 {
            color: #2E7D32;
        }
        .speak-btn {
            background-color: #2E7D32; /* Green color */
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        .speak-btn:hover {
            background-color: #1B5E56; /* Darker green on hover */
        }
        .mute-btn {
            background-color: #FF5722; /* Orange color */
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        .mute-btn:hover {
            background-color: #E64A19; /* Darker orange on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="header">
        <h1> Chat with Dexter 🤖</h1>
        <p>Upload your documents, process them, and ask questions directly from the content.</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state attributes
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'animation_done' not in st.session_state:
        st.session_state.animation_done = False
    if 'mute' not in st.session_state:
        st.session_state.mute = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = ""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load Lottie animation for Dexter from a JSON file
    dexter_animation = load_lottie_file("2.json")

    # Authentication UI
    if not st.session_state.authenticated:
        auth_choice = st.sidebar.selectbox("Choose Authentication", ["Login", "Register"])

        if auth_choice == "Register":
            st.subheader("Register")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Register"):
                response = register_user(email, password)
                if response.status_code == 200:  # Assuming 200 for success
                    st.success("User registered successfully!")
                else:
                    st.error("Registration failed. Try a different email.")

        if auth_choice == "Login":
            st.subheader("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                response = login_user(email, password)
                if response.status_code == 200:
                    st.success("Login successful!")
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    # Introduce Dexter after login
                    if dexter_animation is not None:
                        introduce_dexter(dexter_animation)
                    else:
                        st.error("Failed to load Dexter's animation. Please check the file path.")
                else:
                    st.error("Login failed. Please check your credentials.")
    else:
        # Main Application UI after login
        with st.sidebar.expander("Upload your documents here upload files below 3MB"):
            documents = st.file_uploader("Upload your documents (PDF, DOCX, XLSX)", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)
            if st.button("Process Files"):
                with st.spinner("Processing..."):
                    raw_text = ""
                    if documents:
                        for doc in documents:
                            if doc.name.endswith(".pdf"):
                                raw_text += get_pdf_text([doc])
                            elif doc.name.endswith(".docx"):
                                raw_text += get_docx_text([doc])
                            elif doc.name.endswith(".xlsx"):
                                raw_text += get_excel_text([doc])
                    else:
                        st.error("Please upload a valid document.")
                        return

                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Documents processed and vector store created.")

        # Chat interface
        st.header("Chat with Dexter")
        user_question = st.text_input("Ask a question")

        if st.button("Send"):
            if user_question:
                answer = user_input(user_question)
                st.session_state.chat_history.append({"question": user_question, "answer": answer})

                st.write(f"**Question:** {user_question}")
                st.write(f"**Answer:** {answer}")

                # Optionally, text-to-speech functionality
                if not st.session_state.mute:
                    text_to_speech(answer)
            else:
                st.error("Please enter a question.")

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for chat in st.session_state.chat_history:
                st.write(f"**Question:** {chat['question']}")
                st.write(f"**Answer:** {chat['answer']}")
                st.write("---")

        # Speech to text functionality
        if st.button("Use Voice Command"):
            command = get_voice_command()
            if command:
                st.text_input("Voice Command", value=command, key=uuid.uuid4())
                if command:
                    answer = user_input(command)
                    st.session_state.chat_history.append({"question": command, "answer": answer})
                    st.write(f"**Question:** {command}")
                    st.write(f"**Answer:** {answer}")

                    # Optionally, text-to-speech functionality
                    if not st.session_state.mute:
                        text_to_speech(answer)

        # Mute button
        mute_button_label = "Unmute" if st.session_state.mute else "Mute"
        if st.button(mute_button_label):
            st.session_state.mute = not st.session_state.mute
            st.success(f"Audio {'muted' if st.session_state.mute else 'unmuted'}.")

if __name__ == "__main__":
    main()
