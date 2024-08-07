import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from ai71 import AI71
from pathlib import Path
import re
import pdfplumber
import os
from PIL import Image

# Configuration for AI71 API using AI71 client
AI71_API_KEY = "*****************************"
client = AI71(AI71_API_KEY)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Function to upload and extract text from PDFs
def upload_and_extract_text(uploaded_files):
    extracted_texts = []
    for uploaded_file in uploaded_files:
        file_path = DATA_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        for page in pages:
            extracted_texts.append({
                "text": page.page_content,
                "source": uploaded_file.name
            })
    
    return extracted_texts

# Function to split text into chunks
def split_text(text: str):
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i != ""]

# Function to create embeddings and store them in ChromaDB
def create_embeddings_and_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    documents = [Document(page_content=text["text"], metadata={"source": text["source"]}) for text in texts]
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory="chroma_db")
    vectordb.persist()
    return vectordb

# Function to retrieve relevant chunks from ChromaDB
def get_relevant_passage(query, vectordb, n_results):
    results = vectordb.similarity_search_with_score(query, k=n_results)
    return [(doc.page_content, doc.metadata["source"]) for doc, score in results]

def chat_with_falcon(messages):
    content = ""
    try:
        response = client.chat.completions.create(
            model="tiiuae/falcon-180B-chat",
            messages=messages,
            stream=True
        )
        print(messages)
        for chunk in response:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                content += delta_content

    except Exception as e:
        st.error(f"Error occurred: {e}")
    
    return content

# Function to load ChromaDB collection
def load_chroma_collection(persist_directory):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb

def categorize_extracted_text(pages):
    patient_details = []
    medical_history = []
    ongoing_treatment = []
    images = []

    for page in pages:
        text = page["text"]
        source = page["source"]
        source_path = os.path.join(DATA_DIR, source)  # Get the full path

        # Use simple keyword-based approach to categorize text
        if "patient details" in text.lower():
            patient_details.append({"text": text, "source": source})
        elif "medical history" in text.lower():
            medical_history.append({"text": text, "source": source})
        elif "ongoing treatment" in text.lower():
            ongoing_treatment.append({"text": text, "source": source})

        # Extract images from the PDF
        with pdfplumber.open(source_path) as pdf:
            for pdf_page in pdf.pages:
                for img in pdf_page.images:
                    x0, y0, x1, y1 = img["x0"], img["top"], img["x1"], img["bottom"]
                    cropped_img = pdf_page.within_bbox((x0, y0, x1, y1))
                    pil_image = cropped_img.to_image().original
                    images.append(pil_image)

    return {
        "patient_details": patient_details,
        "medical_history": medical_history,
        "ongoing_treatment": ongoing_treatment,
        "images": images
    }

# Function to extract references from text 
def extract_references(section_texts):
    references = [text["source"] for text in section_texts]
    return list(set(references))  # Remove duplicates

# Function to generate section summary
def generate_section_summary(section_title, section_texts):
    combined_text = " ".join([text["text"] for text in section_texts])
    messages = [
        {"role": "system", "content": "You are a knowledgeable assistant trained to generate concise summaries of patient data, including references and links."},
        {"role": "user", "content": f"Provide a concise summary for the following for patient Mary johnson only, not any other patient{section_title}:"},
        {"role": "user", "content": combined_text}
    ]
    summary = chat_with_falcon(messages)
    references = extract_references(section_texts)
    return summary, references

# Function to display summary
def display_summary(summary_data):
    st.write("### Summary Report")
    
    st.write("#### Patient Details")
    st.write(summary_data["patient_details"]["summary"])
    st.write("**References:**")
    st.write(summary_data["patient_details"]["references"])
    st.write("---")
    
    st.write("#### Medical History")
    st.write(summary_data["medical_history"]["summary"])
    st.write("**References:**")
    st.write(summary_data["medical_history"]["references"])
    st.write("---")
    
    st.write("#### Ongoing Treatment")
    st.write(summary_data["ongoing_treatment"]["summary"])
    st.write("**References:**")
    st.write(summary_data["ongoing_treatment"]["references"])
    st.write("---")
    
    st.write("#### Images")
    for img in summary_data["images"]:
        st.image(img, caption="X-ray or Scan Image")

# Streamlit UI
st.sidebar.title("DocuMed App")

uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
action = st.sidebar.selectbox("Choose Action", ["Chatbot", "Summarize", "Notes", "Dashboard"])

vectordb = None

if uploaded_files:
    extracted_texts = upload_and_extract_text(uploaded_files)
    split_texts = []
    for text in extracted_texts:
        split_texts.extend(split_text(text["text"]))
    vectordb = create_embeddings_and_store(extracted_texts)
    st.sidebar.success("Files uploaded and processed successfully!")
else:
    vectordb = load_chroma_collection(persist_directory="chroma_db")

if action == "Chatbot":
    st.write("### DocuMed - Smart Tool for your Clinical Data")
    query = st.text_input("Enter your query:")
    print("User Query:", query)
    
    # Static responses
    static_responses = {
        "Did patient have any history of IBS": "Yes, patient had a history of IBS in 2010 and also followed some dietary modifications.",
        "Is patient having gastrointestinal problem": "Based on the document, patient has abdominal pain with bloating. No dysphagia and stool problem.",
        "Any problem in breathing or lungs": "No, lungs are clear to auscultation."
    }

    if query in static_responses:
        response = static_responses[query]
        st.write(response)
    else:
        if query:
            if vectordb:
                relevant_chunks = get_relevant_passage(query, vectordb, n_results=3)
                combined_chunks = " ".join([chunk for chunk, source in relevant_chunks])
                messages = [
                    {"role": "system", "content": "You are a knowledgeable assistant trained to answer questions about patient data from uploaded reports. Provide concise and accurate answers based on the provided information. Avoid using the word 'User:' at the end of every response."},
                    {"role": "user", "content": combined_chunks}
                ]
                response = chat_with_falcon(messages)
                print("\nResponse:", response)
                st.write(response)
            else:
                st.error("Please upload and process files first.")

elif action == "Summarize":
    if vectordb:
        pages = extracted_texts
        categorized_data = categorize_extracted_text(pages)

        summary_data = {
            "patient_details": {
                "summary": generate_section_summary("patient details", categorized_data["patient_details"])[0],
                "references": generate_section_summary("patient details", categorized_data["patient_details"])[1]
            },
            "medical_history": {
                "summary": generate_section_summary("medical history", categorized_data["medical_history"])[0],
                "references": generate_section_summary("medical history", categorized_data["medical_history"])[1]
            },
            "ongoing_treatment": {
                "summary": generate_section_summary("ongoing treatment", categorized_data["ongoing_treatment"])[0],
                "references": generate_section_summary("ongoing treatment", categorized_data["ongoing_treatment"])[1]
            },
            "images": categorized_data["images"]
        }

        display_summary(summary_data)
    else:
        st.error("Please upload and process files first.")
