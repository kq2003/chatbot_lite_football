from flask import Flask, request, jsonify
from flask import render_template
from flask_cors import CORS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# create the csv object as a pandas dataframe
import pandas as pd
df = pd.read_csv('FM2023.csv')

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Define the OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'
openai_api_key = os.getenv("OPENAI_API_KEY", "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")

# Define the docsearch the chatbot uses
embeddings = OpenAIEmbeddings()
docsearch = Chroma(persist_directory='embedding', embedding_function=embeddings)

# Define the model and the conversational chatbot (with memory)
llm = OpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=1.2)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, docsearch.as_retriever(), memory=memory)


@app.route('/ask', methods=['POST'])
def ask():
    # Get user input
    data = request.get_json()
    user_input = data['question']

    # If no input is provided, return a 400 error
    if not user_input:
        return jsonify({'error': 'No input provided.'}), 400

    # Otherwise, get the chatbot's response and return it
    result = qa({"question": user_input})
    answer = result['answer']
    # print(result['source_documents'])
    return jsonify(answer)


@app.route('/')
def root():
    return render_template("front_chat.html")



if __name__ == '__main__':
    app.run(debug=True)

