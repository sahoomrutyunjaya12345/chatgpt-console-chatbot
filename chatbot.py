# prompt: make a streamlit file so that i can run on console
import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Step 1: Web scraping (same as before)
url = 'https://botpenguin.com/'
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    website_content = ' '.join([p.text for p in paragraphs])

# Creating a pipeline for question answering (same as before)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def chatbot_response(user_question):
  """
  This function takes a user's question and returns a response 
  based on the website content using the QA pipeline.
  """

  best_answer = qa_pipeline({
      'question': user_question,
      'context': website_content
  })

  return best_answer['answer']

# Streamlit app
st.title("BotPenguin Chatbot")
st.write("Ask me anything about BotPenguin")

user_question = st.text_input("Enter your question here:")

if user_question:
  response = chatbot_response(user_question)
  st.write("Chatbot:", response)

