import os
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# This line tells Flask exactly where to find your HTML
base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(base_dir, 'templates'))

# Global for AI logic
qa_chain = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global qa_chain
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "Missing OPENAI_API_KEY in Vercel Environment Variables"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    temp_path = os.path.join("/tmp", "doc.pdf")
    file.save(temp_path)

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        vectorstore = DocArrayInMemorySearch.from_documents(chunks, OpenAIEmbeddings(openai_api_key=api_key))
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        return jsonify({"message": "Ready to chat!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_chain:
        return jsonify({"error": "Upload PDF first"}), 400
    question = request.json.get("question")
    res = qa_chain.invoke(question)
    return jsonify({"answer": res["result"]})

# This is what Vercel looks for
app_handler = app
