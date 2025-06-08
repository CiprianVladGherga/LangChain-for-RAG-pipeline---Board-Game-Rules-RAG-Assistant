import argparse
import sys
from typing import List, Tuple
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from tqdm import tqdm
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful board game rules assistant. Answer the question based only on the following context:

{context}

---

Question: {question}

Instructions:
1. Answer concisely and accurately
2. If the information is not in the context, say "I cannot find that information in the rulebooks"
3. Include relevant page numbers or rule references when possible
4. Format your response in a clear, easy-to-read manner
"""

def format_response(response_text: str, sources: List[str]) -> str:
    """Format the response with sources and styling."""
    formatted_sources = "\n".join([f"- {source}" for source in sources])
    return f"""
📚 Response:
{response_text}

📖 Sources:
{formatted_sources}
"""

def query_rag(query_text: str) -> Tuple[str, List[str]]:
    """Query the RAG system and return the response with sources."""
    try:
        # Prepare the DB
        print("🔍 Searching rulebooks...", end="", flush=True)
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB
        results = db.similarity_search_with_score(query_text, k=5)
        print(" Done!")

        # Prepare context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Generate response
        print("🤖 Generating response...", end="", flush=True)
        model = Ollama(model="mistral")
        response_text = model.invoke(prompt)
        print(" Done!")

        # Get sources
        sources = [doc.metadata.get("id", "Unknown source") for doc, _score in results]
        
        return response_text, sources

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

def main():
    # Create CLI
    parser = argparse.ArgumentParser(description="Board Game Rules Assistant")
    parser.add_argument("query_text", type=str, help="The question about board game rules")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    args = parser.parse_args()

    # Query the system
    response_text, sources = query_rag(args.query_text)
    
    # Print formatted response
    print(format_response(response_text, sources))

if __name__ == "__main__":
    main()
