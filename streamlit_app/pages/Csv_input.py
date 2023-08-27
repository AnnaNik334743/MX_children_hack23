import streamlit as st
import base64
import pandas as pd
import requests
from io import StringIO
import json

# Streamlit app title and description
st.title("CSV")
st.write("Загрузите файл .csv в формате: ...")

# Upload CSV file
uploaded_file = st.file_uploader("Загрузить csv файл.", type=["csv"])

# API endpoint
api_endpoint = 'http://192.168.83.231:8000/post-csv-streamlit/'


# Function to generate a download link for a DataFrame as a CSV file
def get_csv_download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


# Function to make a POST request and get the response
def make_post_request(df):
    response = requests.post(api_endpoint, files={'input_file': df})
    print(response.content)
    return json.loads(response.content)


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')

    # Button to make the POST request and get response
    if st.button("Обработать файл"):
        response_data = make_post_request(df.to_csv(index=False))
        response_df = pd.read_csv(StringIO(response_data))

        st.write("Обработанный файл:")
        st.write(response_df)

        # Download button for the response CSV
        csv_download_link = get_csv_download_link(response_df, "updated_csv.csv", "Загрузить обработанный файл")
        st.markdown(csv_download_link, unsafe_allow_html=True)
