import os
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# CRITICAL FIX: Explicitly set the template folder path for Vercel
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)

# Pull API Key from Vercel Environment Variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Knowledge base storage
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
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Vercel uses /tmp for temporary file writing
    temp_path = os.path.join("/tmp", file.filename)
    file.save(temp_path)

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({"message": f"Successfully indexed {file.filename}!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    if not qa_chain:
        return jsonify({"error": "Upload a PDF first!"}), 400
    
    data = request.json
    question = data.get("question")
    
    try:
        response = qa_chain.invoke(question)
        return jsonify({"answer": response["result"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Required for local testing
if __name__ == "__main__":
    app.run(debug=True)
