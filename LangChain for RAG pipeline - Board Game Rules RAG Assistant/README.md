# Board Game Rules RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) application that helps users quickly find answers to questions about board game rules. This application uses advanced AI techniques to provide accurate and context-aware responses based on game rulebooks.

## Author
Ciprian Vlad Gherga

## Features
- PDF document processing for board game rulebooks
- Semantic search capabilities using AWS Bedrock embeddings
- Local LLM integration with Ollama (Mistral model)
- Vector database storage using Chroma
- Automated testing suite for rule verification
- Support for multiple board games

## Technical Stack
- Python 3.x
- LangChain for RAG pipeline
- AWS Bedrock for embeddings
- Ollama for local LLM inference
- ChromaDB for vector storage
- PyPDF for document processing

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up AWS credentials:
   - Configure AWS CLI with your credentials
   - Ensure you have access to AWS Bedrock services

3. Install Ollama:
   - Download and install Ollama from https://ollama.ai
   - Pull the Mistral model: `ollama pull mistral`

4. Prepare your data:
   - Place board game rulebook PDFs in the `data` directory
   - Supported format: PDF files

5. Initialize the database:
```bash
python populate_database.py
```

## Usage Examples

### Basic Query
```bash
python query_data.py "How much money do players start with in Monopoly?"
```

### Reset Database
```bash
python populate_database.py --reset
```

### Run Tests
```bash
pytest test_rag.py
```

## Example Applications

1. **Game Rules Assistant**
   - Quick reference for complex board game rules
   - Clarification of ambiguous rule scenarios
   - Learning new games efficiently

2. **Tournament Preparation**
   - Rule verification for tournament organizers
   - Quick fact-checking during competitions
   - Standardized rule interpretations

3. **Game Design Reference**
   - Analysis of existing game mechanics
   - Comparison of different game systems
   - Inspiration for new game designs

4. **Educational Tool**
   - Teaching board game concepts
   - Understanding game mechanics
   - Learning strategy and tactics

## How It Works

1. **Document Processing**
   - PDF rulebooks are loaded and split into manageable chunks
   - Text is processed and cleaned for optimal retrieval

2. **Vector Storage**
   - Text chunks are converted into embeddings using AWS Bedrock
   - Embeddings are stored in ChromaDB for efficient retrieval

3. **Query Processing**
   - User questions are converted to embeddings
   - Similar chunks are retrieved from the database
   - Context is provided to the LLM for accurate responses

4. **Response Generation**
   - Mistral model generates responses based on retrieved context
   - Sources are provided for verification
   - Responses are formatted for easy reading

## Contributing
Feel free to submit issues and enhancement requests!

## License
MIT License
