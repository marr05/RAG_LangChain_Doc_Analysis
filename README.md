# RAG Chat System with LangChain and Ollama

A modern, dark-themed web application for chatting with your PDF documents using Retrieval-Augmented Generation (RAG) powered by LangChain and Ollama.

## Features

- 🌙 **Modern Dark UI** - Sleek, glassmorphic design with purple accents
- 📄 **PDF Upload** - Drag-and-drop or click to upload PDF documents
- 💬 **Interactive Chat** - Ask questions about your uploaded documents
- 🔍 **Smart Retrieval** - Uses ChromaDB for efficient vector search
- 🧠 **Local LLM** - Runs completely offline with Ollama
- 📊 **Statistics** - Track documents and chunks in your database
- 🗑️ **Database Management** - Clear function with auto-cleanup on exit
- 🚀 **Fast & Responsive** - Optimized for performance

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed and running
- At least 8GB RAM (16GB recommended)
- macOS, Linux, or Windows

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-chat-system.git
cd rag-chat-system
```

### 2. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai/download)

### 3. Pull the Llama Model

```bash
# Start Ollama service (if not running)
ollama serve

# In another terminal, pull the model
ollama pull llama3.2
```

### 4. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your settings
nano .env  # or use any text editor
```

**Important configurations in `.env`:**
- `FLASK_SECRET_KEY`: Generate a secure key (see Security section)
- `ADMIN_PASSWORD`: Change from default
- `PORT`: Change if 5000 is in use

## Usage

### Starting the Application

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, start the Flask app
python app.py
```

The application will be available at `http://localhost:3000` (or your configured port [from the main function in app.py]).

### Using the Application

1. **Upload PDFs**: Drag and drop PDF files onto the upload area
2. **Ask Questions**: Type questions in the chat interface
3. **View Sources**: Each answer includes source references
4. **Manage Database**: Use the "Clear Database" button to reset

### Stopping the Application

Press `Ctrl+C` to stop the server. The application will automatically clean up uploaded files.

## Project Structure

```
rag-chat-system/
├── app.py                 # Main Flask application
├── embeddings.py          # HuggingFace embeddings setup
├── populate_db.py         # Database population utilities -> Functionality already embedded in app.py
├── query_data.py          # Query processing utilities    -> Functionality already embedded in app.py
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
├── README.md             # This file
├── templates/            # HTML templates
│   └── index.html        # Main web interface
├── chroma_db/            # Vector database (auto-created)
├── data/                 # Uploaded PDFs (auto-created)
└── uploads/              # Temporary uploads (auto-created)
```

## Security Considerations

### Generate a Secure Secret Key

```python
# Run this to generate a secure key
python -c "import secrets; print(secrets.token_hex(32))"
```

### Best Practices

1. **Never commit `.env`** 
2. **Change default passwords** - Update `ADMIN_PASSWORD`
3. **Use HTTPS in production** - See deployment section
4. **Limit file uploads** - Configure max file size if needed
5. **Regular backups** - Back up your vector database

## Customization

### Change the LLM Model

Edit `.env`:
```
OLLAMA_MODEL=llama2  # or any model from ollama list
```

### Adjust Chunk Size

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for longer chunks
    chunk_overlap=100,
)
```

### Modify UI Theme

Edit color variables in `templates/index.html`:
```css
:root {
    --accent-primary: #6366f1;  /* Change primary color */
    --accent-secondary: #8b5cf6; /* Change secondary color */
}
```

## Troubleshooting

### Common Issues

**Ollama Connection Error:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

**Port Already in Use:**
```bash
# Change port in .env
PORT=8080
```

**Out of Memory:**
- Reduce `chunk_size` in `populate_db.py`
- Use a smaller model (e.g., `ollama pull llama2:7b`)
- Close other applications

**Slow Responses:**
- Ensure Ollama has GPU access (if available)
- Reduce `top_k` parameter in queries
- Use SSD for database storage

## Performance Optimization

1. **GPU Acceleration** (NVIDIA only):
   ```bash
   # Check GPU support
   nvidia-smi
   
   # Ollama automatically uses GPU if available
   ```

2. **Adjust Token Limits**:
   ```python
   # In app.py, modify the ChatOllama parameters
   llm = ChatOllama(
       model="llama3.2",
       num_predict=500,  # Reduce for faster responses
   )
   ```

3. **Database Optimization**:
   - Store database on SSD
   - Regularly clean unused documents
   - Monitor database size## Support

---