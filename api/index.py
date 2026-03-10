import os
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# FIX: Vercel pathing for templates
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '..', 'templates')
app = Flask(__name__, template_folder=template_dir)

# Global for the AI chain
qa_chain = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global qa_chain
    # SAFETY CHECK: If no API key, return a clear error instead of crashing
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({"error": "API Key is missing in Vercel Settings!"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    temp_path = os.path.join("/tmp", "doc.pdf")
    file.save(temp_path)

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        vectorstore = DocArrayInMemorySearch.from_documents(chunks, OpenAIEmbeddings())

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        return jsonify({"message": "File analyzed!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_chain:
        return jsonify({"error": "Upload PDF first"}), 400
    
    question = request.json.get("question")
    res = qa_chain.invoke(question)
    return jsonify({"answer": res["result"]})

# This must match what Vercel expects
app_handler = app
