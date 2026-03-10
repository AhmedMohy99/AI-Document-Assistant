import os
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Simple pathing: templates folder is in the same directory as index.py
app = Flask(__name__)

# Knowledge base global
qa_chain = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global qa_chain
    # CRITICAL: Prevent crash if API key is missing
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "Config Error: OPENAI_API_KEY is not set in Vercel Settings."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    # /tmp is the only writable area on Vercel
    path = os.path.join("/tmp", "doc.pdf")
    file.save(path)

    try:
        loader = PyPDFLoader(path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)
        
        # Initialize AI
        vectorstore = DocArrayInMemorySearch.from_documents(docs, OpenAIEmbeddings(openai_api_key=api_key))
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        return jsonify({"message": "Document indexed successfully!"})
    except Exception as e:
        return jsonify({"error": f"AI Error: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_chain:
        return jsonify({"error": "Please upload a document first."}), 400
    question = request.json.get("question")
    res = qa_chain.invoke(question)
    return jsonify({"answer": res["result"]})

# Required for Vercel
app_handler = app
