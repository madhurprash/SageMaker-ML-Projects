import streamlit as st
import requests
import io
from PyPDF2 import PdfReader
import pandas as pd

st.title("AWS Summarizer")
st.write("This app allows you to upload a CSV or PDF Transcript file, or enter text and ask questions related to the content. The app uses a fine-tuned BART model hosted on SageMaker to summarize and organize your meeting notes.")

AWS_API_GATEWAY_ENDPOINT = "https://n16uhtgob1.execute-api.us-east-1.amazonaws.com/TEST/predicttranscript"

messages = [
    {"role": "system", "content": "You are a professional Question and Answer AI Assistant helping with information in regards to a csv, pdf, and text input file."},
]

def chatbot(query_text, file_data):
    if query_text:
        messages.append({"role": "user", "content": query_text})
    if file_data:
        messages.append({"role": "user", "content": f"{file_type} File Type: {file_data}"})
    
    response = requests.post(AWS_API_GATEWAY_ENDPOINT, json={"messages": messages})
    response_data = response.json()
    
    # Check if 'response' key exists in the response_data
    if "response" in response_data:
        response_text = response_data["response"]
        st.write("Response: " + response_text)
        messages.append({"role": "assistant", "content": response_text})
    else:
        st.write("Error: No response received from the API.")

query_text = st.text_area("Enter Meeting Transcript", height=100)
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
st.markdown("")
st.markdown("<p style='text-align: center'><a href='https://github.com/madhurprash'>Github</a> | <a href='https://www.linkedin.com/in/madhur-prashant-781548179?original_referer=https%3A%2F%2Fwww.google.com%2F'>LinkedIn</a></p>", unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
