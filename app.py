import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import sqlite3

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini Pro model with memory
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Function to get responses from Gemini Pro
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Function to initialize the SQLite database for storing chat history
def initialize_db():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            message TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

# Function to store the conversation in the database
def store_conversation(cursor, role, message):
    cursor.execute('''
        INSERT INTO conversations (role, message)
        VALUES (?, ?)
    ''', (role, message))
    conn.commit()

# Initialize database connection
conn, cursor = initialize_db()

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Chatbot with Memory")

st.header("Chat with AI")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Text input for user question
input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")
clear_response = st.button("Clear Response")  # Clear response button

# If submit button is clicked and there's input
if submit and input:
    # Store user input in session state and database
    st.session_state['chat_history'].append(("You", input))
    store_conversation(cursor, "You", input)

    # Get response from Gemini Pro
    response = get_gemini_response(input)
    st.subheader("The Response is")
    
    # Display and store the bot response
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))
        store_conversation(cursor, "Bot", chunk.text)
        
# Clear response logic
if clear_response:
    # Reset the input field and remove previous chat response from display
    input = ""
    st.session_state['chat_history'].clear()
st.subheader("The Chat History is")

# Display the chat history
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")

# Function to retrieve all conversations from the database
#def get_all_conversations(cursor):
    #cursor.execute('SELECT role, message FROM conversations')
    #return cursor.fetchall()

# Display stored conversation from the database
#st.subheader("Stored Conversation History")
#stored_conversations = get_all_conversations(cursor)
#for role, text in stored_conversations:
    #st.write(f"{role}: {text}")

# Close database connection
def close_connection():
    conn.close()

close_connection()
