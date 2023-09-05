import streamlit as st
import requests
import io
from PyPDF2 import PdfReader
import pandas as pd
import json

st.title("Yoga For Physical and Mental Health Bot")
st.write("This bot provides information and guidance on yoga exercises and practices for improving physical and mental health.")

# SageMaker endpoint URL
SAGEMAKER_ENDPOINT = "https://bedrockForYoga"

messages = [
    {"role": "system", "content": "You are a professional Yoga and Health Assistant providing information and guidance on yoga exercises and practices for physical and mental health."},
]

def chatbot(query_text, file_data):
    if query_text:
        messages.append({"role": "user", "content": query_text})
       
    if file_data:
        messages.append({"role": "user", "content": f"{file_type} File Type: {file_data}"})
    
    # Prepare the request payload for SageMaker
    sagemaker_payload = {
        "query": query_text if query_text else "",  # Use the user's query as the prompt
    }
    
    response = requests.post(SAGEMAKER_ENDPOINT, json=sagemaker_payload, headers={"Content-Type": "application/json"})
    
    if response.status_code == 200:
        response_data = response.json()
        response_text = response_data.get("answer", "No response received from the model.")
        st.write("Response: " + response_text)
        messages.append({"role": "assistant", "content": response_text})
    else:
        st.write("Error: Failed to get a response from the model.")

query_text = st.text_area("Ask a Yoga or Health-related Question", height=100)
file_type = st.selectbox("Select File Type", options=["CSV", "PDF", "Text"])

file_data = None

if file_type == "CSV":
    file = st.file_uploader("Upload CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
        st.write("Uploaded CSV file:")
        st.write(df)
        file_data = df.to_csv(index=False)
elif file_type == "PDF":
    file = st.file_uploader("Upload PDF file", type="pdf")
    if file:
        pdf_reader = PdfReader(file)
        file_data = ""
        for page in pdf_reader.pages:
            file_data += page.extract_text()

        st.write("Uploaded PDF file:")
        with st.container():
            st.markdown(
                "<style>"
                ".scrollable {"
                "    max-height: 300px;"
                "    overflow-y: auto;"
                "}"
                "</style>"
                '<div class="scrollable">'
                + file_data.replace("\n", "<br>")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")
else:
    file_data = st.text_area("Enter text here")

if st.button("Send"):
    try:
        chatbot(query_text, file_data)
    except Exception as e:
        st.error(str(e))

st.markdown("")
st.markdown("---")
st.markdown("<p style='text-align: center'><a href='https://github.com/madhurprash'>Github</a> | <a href='https://www.linkedin.com/in/madhur-prashant-781548179?original_referer=https%3A%2F%2Fwww.google.com%2F'>LinkedIn</a></p>", unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
