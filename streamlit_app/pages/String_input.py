import json

import streamlit as st
import requests


# API endpoint
api_endpoint = 'http://192.168.83.231:8000/post-string/'


# Streamlit app title and description
st.title("Ввод стринги")

def make_post_request(input_string, input_topn):
    response = requests.post(api_endpoint, json={'input_string': input_string,
                                                 'input_topn': input_topn})
    return response.json()

if "my_input" not in st.session_state:
    st.session_state["my_input"] = ""

#
# my_input = st.text_input("Введите адрес:", st.session_state["my_input"])
# input_topn = st.number_input("Количество вариантов:", st.session_state["my_input"], min_value=1, step=1)

input_string = st.text_input("Введите адрес:")
input_topn = st.number_input("Количество вариантов:", min_value=1, step=1)

submit = st.button("Отправить")
if submit:
    if input_string and input_topn:
        st.write("Введенный адрес: ", input_string)
        response = make_post_request(input_string, input_topn)
        st.write("API Response:")
        st.json(json.loads(response))
    else:
        st.write("Please enter both the string and integer.")


