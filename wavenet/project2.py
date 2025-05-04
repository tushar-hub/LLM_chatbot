# %%

import os
import json
import zipfile
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai


#gemini  api key
genai.configure(api_key="AIzaSyAbgDrEtK2hlHE-pcn286ZoInf0H4EVidU")


#  Paths 

BASE_DIR = os.getcwd()
DATA_ZIP = os.path.join(BASE_DIR, "data.zip")
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_db")


#  Dataset Extraction & Parsing

if not os.path.isdir(DATA_DIR):
    if not os.path.isfile(DATA_ZIP):
        raise FileNotFoundError(f"Dataset '{DATA_ZIP}' not found.")
    with zipfile.ZipFile(DATA_ZIP, 'r') as z:
        z.extractall(BASE_DIR)
    print(f"Extracted '{DATA_ZIP}' into '{BASE_DIR}'.")

# Finding json files
json_files = []
for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.lower().endswith('.json'):
            json_files.append(os.path.join(root, f))
if not json_files:
    raise RuntimeError("No JSON files found in data directory.")
print(f"Found {len(json_files)} JSON files.")

# Parse schemes
schemes = []
for path in json_files:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    content = raw.get('data', raw) if isinstance(raw, dict) else {}
    schemes.append({
        'scheme_name': content.get('scheme_name', ''),
        'benefits': content.get('benefits', ''),
        'eligibility': content.get('eligibility', ''),
        'conditions': content.get('conditions', ''),
        'faq': content.get('faq', ''),
        'website': content.get('website', '')
    })
if not schemes:
    raise RuntimeError("Parsed 0 schemes. Check JSON structure.")
print(f"Parsed {len(schemes)} schemes.")


# Chunking & Embedding

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = []
for s in schemes:
    text = (
        f"Scheme Name: {s['scheme_name']}\n"
        f"Benefits: {s['benefits']}\n"
        f"Eligibility: {s['eligibility']}\n"
        f"Conditions: {s['conditions']}\n"
        f"FAQ: {s['faq']}\n"
        f"Website: {s['website']}"
    )
    documents.extend(text_splitter.split_text(text) or [])
if not documents:
    raise RuntimeError("No text chunks generated.")
print(f"Generated {len(documents)} text chunks.")


#  Building  or Load FAISS Index

if not os.path.isdir(VECTOR_DIR):
    db = FAISS.from_texts(documents, embeddings)
    db.save_local(VECTOR_DIR)
    print(f"Saved FAISS index at '{VECTOR_DIR}'.")

db = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()


# Flask App + Embedded Frontend

app = Flask(__name__)
CORS(app)

# Serve embedded HTML UI at root
def html_page():
    return '''<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>MSME Scheme Chatbot</title>
</head>
<body>
  <h1>Scheme Discovery Chatbot</h1>
  <textarea id="query" rows="4" cols="60" placeholder="Ask your question..."></textarea><br>
  <button onclick="sendQuery()">Submit</button>
  <p id="response" style="white-space: pre-wrap;"></p>
  <script>
    async function sendQuery() {
      const q = document.getElementById('query').value;
      const res = await fetch('/query', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ query: q })
      });
      const data = await res.json();
      document.getElementById('response').innerText = data.response || data.error;
    }
  </script>
</body>
</html>'''

@app.route('/')
def serve_index():
    return Response(html_page(), mimetype='text/html')

# RAG endpoint
@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.get_json() or {}
    q = data.get('query', '').strip()
    if not q:
        return jsonify({'error': 'Query required'}), 400

    # Retrieve context chunks
    docs = retriever.get_relevant_documents(q)
    if not docs:
        return jsonify({'response': 'No relevant information found.'})
    context = "\n\n".join([d.page_content for d in docs])

    prompt = (
        "Use the following context to answer the question accurately:"\
        f"\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer:"
    )

   
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat()
    response = chat.send_message(prompt)
    ans = response.text

    return jsonify({'response': ans})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)




# %%
