import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from mistralai import Mistral, UserMessage
import time

# Secure API Key retrieval from Streamlit Secrets
if "MISTRAL_API_KEY" not in st.secrets:
    st.error("API Key not found! Set MISTRAL_API_KEY in Streamlit Secrets before deploying.")
    st.stop()

API_KEY = st.secrets["MISTRAL_API_KEY"]

# --- Functions ---

# Function to scrape policy text
def scrape_policy_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        html_doc = response.text
        soup = BeautifulSoup(html_doc, "html.parser")
        tag = soup.find("div")
        return tag.get_text(strip=True) if tag else None
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve {url}. Error: {e}")
        return None

# Function to generate embeddings with rate limiting and retries
def get_text_embedding(list_txt_chunks, batch_size=20, delay=2):
    client = Mistral(api_key=API_KEY)
    embeddings = []

    for i in range(0, len(list_txt_chunks), batch_size):
        batch = list_txt_chunks[i:i + batch_size]
        retries = 5
        wait_time = delay

        while retries > 0:
            try:
                embeddings_batch_response = client.embeddings.create(
                    model="mistral-embed",
                    inputs=batch
                )
                embeddings.extend([emb.embedding for emb in embeddings_batch_response.data])
                break
            except Exception as e:
                if "429" in str(e):
                    print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2
                    retries -= 1
                else:
                    print(f"Error processing batch: {e}")
                    break

    return embeddings

# Function to classify the intent of the question
def classify_intent(question, policies):
    client = Mistral(api_key=API_KEY)
    prompt = f"""
    Classify the intent of the following question by matching it with the most relevant policy from the list below. Return only the name of the policy.
    Question: {question}
    Policies: {', '.join(policies.keys())}
    """
    messages = [UserMessage(content=prompt)]
    chat_response = client.chat.complete(
        model="mistral-medium-latest",
        messages=messages,
    )
    return chat_response.choices[0].message.content

# Function to query the RAG model
def query_rag_model(question, chunks, embeddings):
    question_embedding = get_text_embedding([question])[0]
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))
    D, I = index.search(np.array([question_embedding]), k=2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """
    return mistral(prompt)

# Function to call Mistral API with retry mechanism
def mistral(user_message, model="mistral-medium-latest"):
    client = Mistral(api_key=API_KEY)
    messages = [UserMessage(content=user_message)]
    
    retries = 5
    wait_time = 5
    while retries > 0:
        try:
            chat_response = client.chat.complete(
                model=model,
                messages=messages,
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2
                retries -= 1
            else:
                raise e


# Define policies
policies = {
     "Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
     "Graduate Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
     "Graduate Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy",
     "Graduate Academic Standing Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-procedure",
     "Graduate Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
     "Graduate Final Grade Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-procedure",
     "Scholarship and Financial Assistance": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
     "International Student Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
     "International Student Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-procedure",
     "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
     "Academic Annual Leave Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
     "Academic Appraisal Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
     "Academic Appraisal Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-procedure",
     "Academic Credentials Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy",
     "Credit Hour Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
     "Intellectual Property Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
     "Examination Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
     "Use Library Space Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy",
     "Library Study Room Booking Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/library-study-room-booking-procedure",
     "Digital Media Centre Booking": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/digital-media-centre-booking"
 }

# Streamlit UI Enhancements
st.markdown("""
    <style>
        .navbar {
            background-color: #b5deff;
            padding: 20px;
            text-align: center;
            font-size: 28px;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .policy-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            padding: 10px;
        }
        .policy-button {
            background-color: #0583D2;
            color: white !important;
            padding: 10px 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 14px;
            font-weight: bold;
            text-decoration: none;
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='navbar'>RAG CHATBOT - 20 UDST POLICIES</div>", unsafe_allow_html=True)

st.markdown("<div class='policy-container'>" + "".join([f"<a href='{url}' class='policy-button'>{policy}</a>" for policy, url in policies.items()]) + "</div>", unsafe_allow_html=True)

st.write("Hey there! This is Parvathy’s Chatbot, your friendly assistant here to help you navigate UDST policies with ease.  Whether you're wondering about graduation requirements, admissions, scholarships, or any other policy, I’ve got your back! Just type your question, and I'll not only classify the intent but also fetch the most relevant policy details for you. ")

# User Query Input
user_query = st.text_area("Enter your question:", placeholder="E.g., What is the graduation policy at UDST?")

# Submit Button
if st.button("Get Answer"):
    if user_query:
        with st.spinner("Processing your query..."):
            # Classify the intent of the question
            intent = classify_intent(user_query, policies)
            st.write(f"Identified Policy: {intent}")
            
            # Scrape and process policy texts
            policy_texts = [scrape_policy_text(url) for url in policies.values() if scrape_policy_text(url)]
            
            # Split text into chunks
            chunk_size = 512
            chunks = [text[i:i + chunk_size] for text in policy_texts for i in range(0, len(text), chunk_size)]
            
            # Generate embeddings
            text_embeddings = get_text_embedding(chunks)

            # Query model for full answer
            answer = query_rag_model(user_query, chunks, text_embeddings)

            st.success("Answer found!")
            with st.expander("View Answer"):
                st.write(answer)
    else:
        st.warning("Please enter a query.")
