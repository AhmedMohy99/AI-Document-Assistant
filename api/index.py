import os
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

app = Flask(__name__, template_folder='../templates')

# Set your API Key here or in environment variables
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Global variable to store the "knowledge base" of the uploaded document
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
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save file temporarily to process it
    temp_path = os.path.join("/tmp", file.filename)
    file.save(temp_path)

    try:
        # 1. Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # 2. Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # 3. Create Vector Store (Knowledge Base)
        embeddings = OpenAIEmbeddings()
        vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)

        # 4. Initialize QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        os.remove(temp_path) # Clean up
        return jsonify({"message": f"Successfully indexed {file.filename}!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    if not qa_chain:
        return jsonify({"error": "Please upload a document first!"}), 400
    
    question = request.json.get("question")
    response = qa_chain.invoke(question)
    return jsonify({"answer": response["result"]})

if __name__ == "__main__":
    app.run(debug=True)
