import os
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# This fix ensures Flask finds your HTML regardless of Vercel's server location
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

# Global variable for the knowledge base
qa_chain = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global qa_chain
    # Safety Check: Check for API Key first
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "Config Error: OPENAI_API_KEY is missing in Vercel Settings."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    # /tmp is the only writable directory on Vercel
    temp_path = os.path.join("/tmp", "document.pdf")
    file.save(temp_path)

    try:
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        docs = splitter.split_documents(pages)
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        return jsonify({"message": "PDF analyzed successfully!"})
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_chain:
        return jsonify({"error": "Please upload a document first."}), 400
    
    question = request.json.get("question")
    try:
        res = qa_chain.invoke(question)
        return jsonify({"answer": res["result"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Required for Vercel to find the application
app_handler = app
