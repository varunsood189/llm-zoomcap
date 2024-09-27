import streamlit as st
import time
from ai21 import AI21Client
from ai21.models.chat import UserMessage
import os
import json
from elasticsearch import Elasticsearch
from openai import OpenAI

AI21_API_KEY = os.environ["AI21_API_KEY"]

client  = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama'
)
es_client = Elasticsearch('http://localhost:9200')

# rag: augmentation 
def build_prompt(query,search_results):
    prompt_template="""
You are a teaching assistant, Please answer the QUESTION based on facts from the CONTEXT, 
If CONTEXT doesnot have the facts, Please answer with NONE.

QUESTION: {question}

CONTEXT: {context}
""".strip()
    context =""
    for doc in search_results:
        context =context+f"section: {doc['section']} \nquestion: {doc["question"]}\n answer: {doc["text"]} \n\n"
    prompt = prompt_template.format(question=query , context=context).strip()
    return prompt 

# rag: generation 
def single_message_instruct(content):
    messages = [UserMessage(content=content)]
    j_client = AI21Client(api_key=AI21_API_KEY)
    response = j_client.chat.completions.create(
        model="jamba-1.5-large",
        messages=messages,
        top_p=1.0 # Setting to 1 encourages different responses each call.
    )
    return response.to_json()
def llm(prompt):
    response = single_message_instruct(prompt)
    json_response = json.loads(response)
    content = json_response["choices"][0]["message"]["content"]
    return content

def elastic_search(query,index_name= "course-questions"):
    search_query= {
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^3", "text", "section"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
    }

    response= es_client.search(index=index_name,body=search_query)
    results_docs =[]
    for hit in response["hits"]["hits"]:
        results_docs.append(hit["_source"])
    return results_docs

# Simulate your RAG function here
def rag(query):
    search_results = elastic_search(query)
    prompt = build_prompt(query,search_results)
    answer = llm(prompt)
    return f"RAG output for '{answer}'"

# Streamlit app
st.title("RAG Application")

# Input box for the query
input_text = st.text_input("Enter your query:")

# Button to submit query
if st.button("Ask"):
    if input_text:
        # Display a spinner while waiting for the function to complete
        with st.spinner('Running RAG, please wait...'):
            # Run the RAG function
            result = rag(input_text)
        # Display the output
        st.success(f"Output: {result}")
    else:
        st.error("Please enter a query")
