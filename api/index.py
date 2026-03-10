import os
from flask import Flask, render_template, request, jsonify

# We use standard PDF logic that is highly stable on Vercel
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Fix for Vercel's directory structure
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '..', 'templates')

app = Flask(__name__, template_folder=template_dir)

# Ensure the app key is read correctly
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

qa_chain = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global qa_chain
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        # Use the /tmp folder - the only writable place on Vercel
        temp_path = os.path.join("/tmp", "uploaded_doc.pdf")
        file.save(temp_path)

        loader = PyPDFLoader(temp_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        docs = text_splitter.split_documents(data)

        # Vector store creation
        embeddings = OpenAIEmbeddings()
        vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        return jsonify({"message": "Document indexed!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    if not qa_chain:
        return jsonify({"error": "No document uploaded"}), 400
    question = request.json.get("question")
    response = qa_chain.invoke(question)
    return jsonify({"answer": response["result"]})

# This is the line Vercel looks for
app_handler = app
