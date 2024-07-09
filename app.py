import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pyttsx3

load_dotenv()

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store using FAISS
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
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed percent (can go over 100)
    engine.setProperty('volume', 1.0)  # Volume 0-1
    engine.say(text)
    engine.runAndWait()

# Main Streamlit application
def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("PDF Chatbot")

    st.markdown(
        """
        <style>
        .main {
            background-color: #000000;
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
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="header">
        <h1> Chat with Dexter 🤖</h1>
        <p>Upload your PDF files, process them, and ask questions directly from the content.</p>
    </div>
    """, unsafe_allow_html=True)

    # File uploader for PDF files
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
    
    if pdf_docs:
        # Process PDF files and store vector store
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete. You can now ask questions!")

        # Function to handle user question input
        def handle_user_question(user_question):
            if user_question:
                with st.spinner("Fetching answer..."):
                    response = user_input(user_question)
                    st.markdown(f"""
                    <div class="response">
                        <h3>Reply:</h3>
                        <p>{response}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    text_to_speech(response)  # Read out the response

        # Input for user question (text input or voice command)
        user_question = st.text_input("Ask a Question from the PDF Files", placeholder="Type your question here...")

        if st.button("Ask Dexter"):
            handle_user_question(user_question)

    # Voice recognition script using webkitSpeechRecognition
    st.markdown("""
    <script>
        var recognition;
        function startRecognition() {
            if (!('webkitSpeechRecognition' in window)) {
                document.getElementById('status').innerHTML = 'Your browser does not support voice recognition.';
                return;
            }

            recognition = new webkitSpeechRecognition();
            recognition.lang = 'en-US';
            recognition.continuous = false;
            recognition.interimResults = false;

            recognition.onstart = function() {
                document.getElementById('status').innerHTML = 'Voice recognition started. Try speaking into the microphone.';
            };

            recognition.onerror = function(event) {
                document.getElementById('status').innerHTML = 'Error occurred in recognition: ' + event.error;
            };

            recognition.onend = function() {
                document.getElementById('status').innerHTML = 'Voice recognition ended.';
            };

            recognition.onresult = function(event) {
                var userQuestion = event.results[0][0].transcript;
                document.getElementById('status').innerHTML = 'You said: ' + userQuestion;
                document.getElementById('user-question').value = userQuestion;
                var askButton = document.getElementById('ask-button');
                askButton.click();  // Simulate click on Ask button after voice input
            };

            recognition.start();
        }
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
