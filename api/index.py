import os
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# FIX: Manually calculate the path to the templates folder
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '..', 'templates')

app = Flask(__name__, template_folder=template_dir)

# Pull key from Vercel Environment Variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Global variable for the AI chain
qa_chain = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global qa_chain
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    # Vercel requires writing to the /tmp directory
    temp_path = os.path.join("/tmp", file.filename)
    file.save(temp_path)

    try:
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

        vectorstore = DocArrayInMemorySearch.from_documents(docs, OpenAIEmbeddings())

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        return jsonify({"message": "File processed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_chain:
        return jsonify({"error": "Please upload a PDF first"}), 400
    
    question = request.json.get("question")
    response = qa_chain.invoke(question)
    return jsonify({"answer": response["result"]})

# Required for Vercel to find the app
app_handler = app
