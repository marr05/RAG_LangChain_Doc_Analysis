import os
import shutil
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
import atexit
import signal
import sys

# Import your existing modules
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema.document import Document
from embeddings import get_embeddings

app = Flask(__name__)

# Generate a secret key for production use
# Option 1: Use environment variable (recommended for production)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', None)

# Option 2: Generate a random key if not set (for development)
if not app.secret_key:
    import secrets
    app.secret_key = secrets.token_hex(32)
    print(f"WARNING: Using generated secret key. For production, set FLASK_SECRET_KEY environment variable.")

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
CHROMA_PATH = "chroma_db"
DATA_PATH = "data"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# Initialize the vector store
vector_store = None

PROMPT_TEMPLATE = """
You are a helpful AI assistant analyzing documents. Use the following context to answer the question thoroughly and comprehensively.

Context:
{context}

---

Instructions:
1. Provide a detailed and complete answer based on the context above
2. Include relevant details, examples, and explanations from the context
3. Structure your response with clear paragraphs when appropriate
4. If the context contains partial information, explain what is available and what might be missing
5. If the answer is not in the context, say "I don't have enough information in the provided documents to answer this question."

Question: {question}

Detailed Answer:"""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_vector_store():
    """Initialize or get the existing vector store."""
    global vector_store
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
    )
    return vector_store

def load_single_pdf(file_path):
    """Load a single PDF file."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def calculate_chunk_ids(chunks):
    """Calculate unique IDs for chunks."""
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata['id'] = chunk_id
        
    return chunks

def add_documents_to_db(file_path):
    """Process and add a PDF to the vector database."""
    try:
        # Load the PDF
        documents = load_single_pdf(file_path)
        
        # Split into chunks
        chunks = split_documents(documents)
        
        # Add chunk IDs
        chunks_with_ids = calculate_chunk_ids(chunks)
        
        # Get existing IDs
        existing_items = vector_store.get(include=[])
        existing_ids = set(existing_items['ids'])
        
        # Filter new chunks
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata['id'] not in existing_ids:
                new_chunks.append(chunk)
        
        # Add new chunks
        if new_chunks:
            new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
            vector_store.add_documents(new_chunks, ids=new_chunk_ids)
            return len(new_chunks)
        
        return 0
    except Exception as e:
        raise e

def query_rag(query_text: str, top_k: int = 5):
    """Query the RAG system with improved response generation."""
    try:
        # Perform similarity search with more context
        docs = vector_store.similarity_search_with_score(query_text, k=top_k)
        
        if not docs:
            return "No relevant documents found in the database. Please upload some PDF documents first.", []
        
        # Prepare context with more information
        context_parts = []
        for i, (doc, score) in enumerate(docs):
            # Add document context with relevance indicator
            context_parts.append(f"[Document {i+1} - Relevance: {1-score:.2f}]\n{doc.page_content}")
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Create prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Get response from LLM with adjusted parameters for longer responses
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.3,  # Slightly higher for more natural responses
            num_predict=1000,  # Increase token limit for longer responses
            max_tokens=1500  # Allow longer responses
        )
        response = llm.invoke(prompt)
        
        # Extract sources with better formatting
        sources = []
        for doc, score in docs:
            source_id = doc.metadata.get("id", "Unknown")
            # Extract filename from source path
            source_file = source_id.split('/')[-1].split(':')[0] if '/' in source_id else source_id.split(':')[0]
            sources.append(f"{source_file} (p.{doc.metadata.get('page', '?')})")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        
        return response.content if hasattr(response, 'content') else str(response), unique_sources[:3]  # Limit to 3 sources
    except Exception as e:
        return f"Error querying the system: {str(e)}", []

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(DATA_PATH, filename)
            
            # Save the file
            file.save(filepath)
            
            # Process and add to vector store
            chunks_added = add_documents_to_db(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'chunks_added': chunks_added,
                'message': f'Successfully processed {filename}. Added {chunks_added} new chunks to the database.'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400

@app.route('/query', methods=['POST'])
def query():
    """Handle chat queries."""
    data = request.get_json()
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        response, sources = query_rag(query_text)
        return jsonify({
            'response': response,
            'sources': sources
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_db', methods=['POST'])
def clear_database():
    """Clear the vector database and optionally the uploaded files."""
    global vector_store
    try:
        # Remove the existing database directory
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            print(f"Removed existing database at {CHROMA_PATH}")
        
        # Create fresh directory
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        # Reinitialize vector store
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embeddings(),
        )
        
        # Clear the data directory to remove uploaded PDFs
        if os.path.exists(DATA_PATH):
            shutil.rmtree(DATA_PATH)
            os.makedirs(DATA_PATH, exist_ok=True)
            print(f"Cleared uploaded files from {DATA_PATH}")
        
        return jsonify({
            'success': True,
            'message': 'Database and uploaded files cleared successfully.'
        })
    except Exception as e:
        print(f"Error clearing database: {str(e)}")
        return jsonify({'error': f'Failed to clear database: {str(e)}'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get database statistics."""
    try:
        items = vector_store.get(include=[])
        
        # Handle empty database case
        if not items or 'ids' not in items or len(items['ids']) == 0:
            return jsonify({
                'num_documents': 0,
                'num_chunks': 0
            })
        
        # Count unique documents
        unique_docs = set()
        for id in items['ids']:
            # Extract document path from the ID
            doc_path = id.split(':')[0] if ':' in id else id
            unique_docs.add(doc_path)
        
        return jsonify({
            'num_documents': len(unique_docs),
            'num_chunks': len(items['ids'])
        })
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return jsonify({
            'num_documents': 0,
            'num_chunks': 0
        })

# Initialize vector store on startup
initialize_vector_store()

# Cleanup function
def cleanup_on_exit():
    """Clean up data folder on application exit."""
    print("\n🧹 Cleaning up temporary files...")
    
    # Clear the data directory
    if os.path.exists(DATA_PATH):
        try:
            shutil.rmtree(DATA_PATH)
            os.makedirs(DATA_PATH, exist_ok=True)
            print(f"✅ Cleared data folder: {DATA_PATH}")
        except Exception as e:
            print(f"❌ Error clearing data folder: {e}")
    
    # Optionally clear the vector database too
    # Uncomment these lines if you want to clear the database on exit as well
    # if os.path.exists(CHROMA_PATH):
    #     try:
    #         shutil.rmtree(CHROMA_PATH)
    #         print(f"✅ Cleared vector database: {CHROMA_PATH}")
    #     except Exception as e:
    #         print(f"❌ Error clearing database: {e}")
    
    print("👋 Cleanup complete. Goodbye!")

# Register cleanup function for different exit scenarios
atexit.register(cleanup_on_exit)

# Handle Ctrl+C gracefully
def signal_handler(sig, frame):
    print('\n⚠️  Interrupt received, shutting down gracefully...')
    cleanup_on_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 3000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('HOST', '0.0.0.0')
    
    print(f"\n🚀 Starting RAG Chat System...")
    print(f"📍 Server running at http://localhost:{port}")
    print(f"📁 Data folder: {DATA_PATH}")
    print(f"🗄️  Database folder: {CHROMA_PATH}")
    print(f"\n💡 Tip: Press Ctrl+C to stop the server and clean up files\n")
    
    try:
        app.run(debug=debug, host=host, port=port)
    except Exception as e:
        print(f"\n❌ Error running server: {e}")
        cleanup_on_exit()