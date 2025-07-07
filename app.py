# Learning LangGraph
# Learning creating Graphs
# Connecting Categories as Node and differentiating categories from user queries based on Intent Classification
# Also using RAG for Document Related Queries

import os
import warnings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Annotated
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from simpleeval import simple_eval

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# Model
set_llm_cache(InMemoryCache())  # Set up caching to avoid repeating LLM calls
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)

# Output Parser
parser = StrOutputParser()

# Prompt
rag_prompt = PromptTemplate(
    template="""
You are an intelligent assistant that only answers questions based on the provided document content.

The document may include:
- Headings, paragraphs, subheadings
- Lists or bullet points
- Tables or structured data
- Text from PDF, DOCX, or TXT formats

Your responsibilities:
1. Use ONLY the content in the document to answer.
2. If the user greets (e.g., "hi", "hello"), respond with a friendly greeting.
3. Otherwise, provide a concise and accurate answer using only the document content.

Document Content:
{context}

User Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# Document Loader
def extract_document(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            return "Unsupported file format. Please upload a PDF, DOCX or TXT file."
        
        docs = loader.load()
        print(f'Loading {file_path} ......')
        print('Loading Successful')
        return docs

    except FileNotFoundError as fe:
        print(f"File not found: {fe}")
    except Exception as e:
        print(f"Error loading document: {e}")
    return []

# RAG Workflow
def create_rag_chain(doc, prompt, parser, score_threshold=1.0, resume_text=False):
    # Document Loader
    docs = extract_document(doc)
    if not docs:
        raise ValueError("Document could not be loaded.")
    # Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Model Embeddings and Vector Store
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(chunks, embedding_model)
    # Retriever for Confidence Score
    retriever = vector_store.similarity_search_with_score

    def retrieve_using_confidence(query):
        results = retriever(query)
        filtered = [doc for doc, score in results if score <= score_threshold]
        return filtered
    
    def format_docs(retrieved_docs):
        if not retrieved_docs:
            return "INSUFFICIENT CONTEXT"
        return "\n\n".join([doc.page_content for doc in retrieved_docs])

    parallel_chain = RunnableParallel({
        'context': RunnableLambda(lambda q: retrieve_using_confidence(q)) | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = parallel_chain | rag_prompt | llm | parser

    if resume_text:
        return main_chain, "\n\n".join([doc.page_content for doc in docs])
    return main_chain

# LangGraph =================================================================================================================>

# State Definition
class State(TypedDict):    # This defines what data the graph will pass between steps.
    question: str   # the user’s input
    category: Literal["math", "general", "help", "greet", "bye"]    # will be either "math", "general" or "some other category"  
    answer: str    # the final result

# Router(Special Graph): Decide if it's math or general question
def router_node(state:State) -> dict:
    question = state.get("question", "")
    # Intent Prompt
    intent_prompt = PromptTemplate(
        template="""You are an AI Intent Classifier for the Jobma Interviewing Platform. Based on the user input, identify their intent from the list of predefined intents.

    Possible Intents:
    - **greet**: The user says hello, hi, good morning, or other greeting-like phrases.
    - **help**: The user is asking for help or support about using the Jobma platform.
    - **bye**: The user says goodbye or ends the conversation.
    - **general**: The user is asking a general question unrelated to the bot itself (e.g., "What is the capital of France?", "Tell me a joke").
    - **math**: The user is asking a math-related question or calculation (e.g., "What is 3 + 5?", "Calculate 10/2", "Solve 2*3+1").

    Classify the following user input strictly as one of the intents above. Your response must be a **single word** from the list: `greet`, `help`, `bye`, `math`, or `general`.

    User Input:
    "{input}"

    Intent:
    """,
        input_variables=['input']
    )
    formatted_prompt = intent_prompt.format(input=question)
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    category = response.content.strip().lower()

    if category not in ['greet', 'help', 'bye', 'general', 'math']:
        category = 'general'    # Fallback
    
    return {'category': category}

# Math Node
def math_node(state: State) -> dict:
    question = state.get("question", "")
    response = llm.invoke([HumanMessage(content=question)])
    return {"answer": response.content}

# RAG Node
def rag_node(state: State) -> dict:
    question = state.get("question", "")
    rag_chain = create_rag_chain("Sensitive_Documents/formatted_QA.txt", rag_prompt, parser)
    answer = rag_chain.invoke(question)

    return {"answer": answer}

# General LLM Node
def general_node(state:State) -> dict:
    question = state['question']
    response = llm.invoke([HumanMessage(content=question)])
    return {"answer": response.content}

# Greet Node
def greet_node(state: State) -> dict:
    question = state.get("question", "")
    response = llm.invoke([HumanMessage(content=question)])
    return {"answer": response.content}

# Bye Node
def bye_node(state: State) -> dict:
    question = state.get("question", "")
    response = llm.invoke([HumanMessage(content=question)])
    return {"answer": response.content}


# Build the LangGraph
builder = StateGraph(State)  # Main Graph

builder.add_node("router", router_node)
builder.add_node("rag_node", rag_node)
builder.add_node("math_node", math_node)
builder.add_node("general_node", general_node)
builder.add_node("greet_node", greet_node)
builder.add_node("bye_node", bye_node)


# Edges
builder.set_entry_point("router")
builder.add_conditional_edges(
    "router",
    lambda state: state['category'],
    {
        "math": "math_node",
        "general": "general_node",
        "help": "rag_node",
        "greet": "greet_node",
        "bye": "bye_node"
    }
)

builder.add_edge("math_node", END)
builder.add_edge("general_node", END)
builder.add_edge("rag_node", END)
builder.add_edge("greet_node", END)
builder.add_edge("bye_node", END)

# Compile the App
graph = builder.compile()

# Visualizing the Graph
print("\nGraph Structure:")
print(graph.get_graph().print_ascii())
print("=======================================================================================")

def ask_ai():
    while(True):
        user_question = input("Ask Something: ")
        
        if user_question == 'exit':
            break

        result = graph.invoke({"question": user_question})
        print("Answer: ", result['answer'])

# Run the App
if __name__ == "__main__":
    ask_ai()