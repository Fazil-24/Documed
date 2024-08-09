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
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Configuration for AI71 API using AI71 client
AI71_API_KEY = "***********************************"
client = AI71(AI71_API_KEY)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

extracted_texts = []

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

# Main function to chat with falcom llm model
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

# Function to categorize text
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
        {"role": "user", "content": f"Provide a concise summary for the following {section_title}:"},
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

# Function to summarize the note using Falcon model
def summarize_note(note):
    messages = [
        {"role": "system", "content": "You are a helpful assistant trained to summarize doctor's notes into concise one-line summaries."},
        {"role": "user", "content": note}
    ]
    summary = ""
    try:
        response = client.chat.completions.create(
            model="tiiuae/falcon-180B-chat",
            messages=messages,
            stream=True
        )
        for chunk in response:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                summary += delta_content
    except Exception as e:
        st.error(f"Error occurred: {e}")
    
    return summary

# Function to add a new note
def add_note():
    date = st.date_input("Date", datetime.now())
    doctor_name = st.text_input("Doctor Name")
    hospital_name = st.text_input("Hospital Name")
    note = st.text_area("Notes")
    
    if st.button("Publish"):
        summarized_note = summarize_note(note)
        new_note = {
            "Date": date.strftime("%Y-%m-%d"),
            "Doctor Name": doctor_name,
            "Hospital Name": hospital_name,
            "Notes": summarized_note
        }
        st.session_state.notes.append(new_note)
        st.success("Note added successfully!")

# Function to display the notes table
def display_notes_table():
    if st.session_state.notes:
        df = pd.DataFrame(st.session_state.notes)
        st.table(df)
    else:
        st.write("No notes available.")


########################################################### UI #####################################################################
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
    st.write("# DocuMed - Smart Tool for Clinical Data ")
    query = st.text_input("Enter your query:")
    print("User Query:", query)
    if query:
        if vectordb:
            relevant_chunks = get_relevant_passage(query, vectordb, n_results=3)
            combined_chunks = " ".join([chunk for chunk, source in relevant_chunks])
            messages = [
                {"role": "system", "content": "You are a knowledgeable assistant trained to answer questions about patient data from uploaded reports. Provide concise and accurate answers based on the provided information. Avoid using the word 'User:' at the end of every response"},
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


elif action == "Notes":
    st.title("Doctor's Notes")
    if 'notes' not in st.session_state:
        st.session_state.notes = []
    add_note()
    st.write("### Notes Table")
    display_notes_table()


elif action == "Dashboard":
    # This section under development, its upcoming , currently trying to grab patient health metrics and display it with dummy values for testin purpose
    def calculate_percentage(result, ref_interval):
        try:
            ref_values = ref_interval.split(' - ')
            ref_min = float(ref_values[0].strip())
            ref_max = float(ref_values[1].strip())
            avg_ref = (ref_min + ref_max) / 2
            percentage = (result / avg_ref) * 100
            return percentage
        except (ValueError, IndexError):
            return None

    def create_individual_charts(df):
        fig_list = []
        
        for index, row in df.iterrows():
            result = float(row["Result"])
            ref_interval = row["Reference Interval"]
            
            percentage = calculate_percentage(result, ref_interval)
            
            if percentage is not None:
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=percentage,
                    title={'text': row["Parameter"], 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "cyan"},
                        'bgcolor': "lightgray",
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                    }
                ))
                fig.update_layout(
                    height=300,
                    width=300,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    font=dict(size=12),
                    annotations=[{
                        'text': "%",
                        'font': {'size': 20},
                        'xref': 'paper', 'yref': 'paper',
                        'x': 0.5, 'y': 0.5,
                        'showarrow': False
                    }],
                    shapes=[{
                        'type': 'rect',
                        'xref': 'paper', 'yref': 'paper',
                        'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
                        'line': {
                            'color': 'white',
                            'width': 2
                        }
                    }]
                )
                fig_list.append(fig)
        
        return fig_list

    def display_dashboard(df):
        st.write("### Lab Test Dashboard")
        
        if df.empty:
            st.write("No data available.")
            return
        
        fig_list = create_individual_charts(df)
        
        if not fig_list:
            st.write("No valid data available for plotting.")
            return
        
        # Displaying charts in a grid
        cols = st.columns(2)
        for i, fig in enumerate(fig_list):
            with cols[i % 2]:
                st.plotly_chart(fig)

    # Example DataFrame based on provided sample
    data = {
        "Parameter": ["Hemoglobin", "RBC Count"],
        "Result": [12, 4.00],
        "Units": ["g/dL", "mill/mm3"],
        "Reference Interval": ["13 - 17", "4.50 - 5.50"]
    }
    df = pd.DataFrame(data)
    # Display the dashboard
    display_dashboard(df)



