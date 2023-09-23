import streamlit as st
import requests
import json

url = "http://127.0.0.1:5000/query"


headers = {
  'Content-Type': 'application/json'
}

st.set_page_config(page_title="IR BOT", page_icon="https://www.merilytics.com/wp-content/uploads/2018/08/cropped-merilytics-fav-icon-192x192.png")

st.title("Information Retrieval Bot")

user_query = st.text_area("What would you like to know?")
if st.button('Submit'):
    with st.spinner("Retrieving information"):
        payload = json.dumps({
            "query": user_query
        })
        response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
        result = response.json()["result"].strip(" ")
        st.text_area(label="Result", value=result, height=100)