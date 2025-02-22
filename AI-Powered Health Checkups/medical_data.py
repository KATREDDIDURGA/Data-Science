import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import pdfplumber
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Medical Report Assistant", page_icon="🩺", layout="wide")

# Function to apply custom CSS for styling
def apply_custom_styles():
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50; 
            border: none;
            color: white;
            padding: 10px 24px;
            text-align: center;
            font-size: 16px;
        }
        .stTitle {
            color: #2c3e50;
        }
        </style>
    """, unsafe_allow_html=True)

# Apply styles
apply_custom_styles()

# Login Page for Patient Authentication
# Login Page for Patient Authentication
def login_page():
    st.title("Patient Login")

    # Display an image at the top of the login page
    st.image("health.png", width =300)

    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Patient information for authentication
    patients = {
        "patient1": {"username": "patient1", "password": "health123"},
        "patient2": {"username": "patient2", "password": "medic456"},
    }

    # Login button
    if st.button("Login"):
        if username in patients and password == patients[username]["password"]:
            st.session_state['authenticated'] = True
            st.session_state['patient'] = username
            st.success(f"Logged in as {username.capitalize()}")
        else:
            st.error("Invalid credentials. Please try again.")


# Medical Report Upload and Extraction
def upload_medical_report():
    st.title("Upload Medical Report")

    # File uploader for PDF medical report
    uploaded_file = st.file_uploader("Upload PDF Report", type="pdf")
    
    if uploaded_file:
        # Display loading message while processing
        with st.spinner("Extracting data from medical report..."):
            text_content, table_data = extract_data_from_pdf(uploaded_file)
            st.session_state['report_text'] = text_content
            st.session_state['report_table'] = table_data
            st.success("Data extracted successfully!")

        # Display extracted text and tables
        st.subheader("Extracted Text")
        st.write(text_content)

        if table_data:
            st.subheader("Extracted Tables")
            st.write(table_data)

        # Generate medical suggestions based on extracted data
        if st.button("Get Suggestions"):
            generate_suggestions(text_content)

# Function to extract text and table data from PDF using pdfplumber
def extract_data_from_pdf(pdf_file):
    text_content = ""
    table_data = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text_content += page.extract_text() + "\n"
            table_data.extend(page.extract_tables())

    # Convert extracted table data to DataFrame if tables are found
    if table_data:
        table_data = [pd.DataFrame(table) for table in table_data]
    return text_content, table_data

# Function to generate medical suggestions based on extracted data
def generate_suggestions(text_content):
    st.subheader("Medical Suggestions")

    # Initialize the LLM with OpenAI API key (Replace with your actual key)
    api_key = "#"
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)
    
    # Split the extracted text into smaller chunks for embedding
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text_content)
    
    # Generate embeddings on-the-fly for the extracted chunks
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    
    # Set up the RetrievalQA chain with the dynamically created retriever
    retriever = knowledge_base.as_retriever()
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    
    # Run the query and get suggestions
    response = qa_chain({"query": f"Suggest treatments and precautions based on: {text_content}"})
    suggestions = response["result"]
    
    st.write(suggestions)

# Main logic for page navigation and authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    login_page()
else:
    # Navigation between uploading medical report and analyzing it
    tab1, tab2 = st.tabs(["Medical Report Upload", "Medical Suggestions"])
    
    with tab1:
        upload_medical_report()
    
    with tab2:
        if 'report_text' in st.session_state:
            generate_suggestions(st.session_state['report_text'])
        else:
            st.write("Please upload a medical report in the 'Medical Report Upload' tab first.")
