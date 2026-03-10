import os
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# FIX: Absolute pathing to ensure index.html is found
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

# Global for the AI chain
qa_chain = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global qa_chain
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY is missing in Vercel Settings"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    path = os.path.join("/tmp", "doc.pdf")
    file.save(path)

    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        vectorstore = DocArrayInMemorySearch.from_documents(chunks, OpenAIEmbeddings(openai_api_key=api_key))
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        return jsonify({"message": "Document indexed!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_chain:
        return jsonify({"error": "Upload a PDF first."}), 400
    question = request.json.get("question")
    res = qa_chain.invoke(question)
    return jsonify({"answer": res["result"]})

# DEBUG ROUTE: If you get a 404, go to /debug to see the server path
@app.route('/debug')
def debug():
    return jsonify({
        "current_dir": os.getcwd(),
        "template_dir": template_dir,
        "templates_exist": os.path.exists(template_dir)
    })

app_handler = app
