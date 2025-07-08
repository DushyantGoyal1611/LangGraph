import os
import json
import warnings
from dotenv import load_dotenv
from datetime import datetime, timedelta
from functools import lru_cache
# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, List, Tuple, Optional, Union
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain.memory import ConversationBufferMemory
# SQL
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# Input Schema Using Pydantic
    # For Interview Scheduling
class ScheduleInterviewInput(BaseModel):
    role:str = Field(description="Target Job Role")
    resume_path:str = Field(description="Path to resume file (PDF/DOCX/TXT)")
    question_limit:int = Field(description="Number of interview questions to generate")
    sender_email:EmailStr = Field(description="Sender's email address")

    # For Tracking Candidate
class TrackCandidateInput(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[EmailStr] = Field(None, description="Email address of the candidate")
    role: Optional[str] = Field(None, description="Role applied for, e.g., 'frontend', 'backend'")
    date_filter: Optional[str] = Field(
        None,
        description="Optional date filter: 'today', 'recent', or 'last_week'"
    )
    status: Optional[Literal["Scheduled", "Completed"]] = None

# SQL Connection
@lru_cache(maxsize=1)
def create_connection():
    print("Creating Connection with DB")
    try:
        user = os.getenv("DB_USER")
        raw_password = os.getenv("DB_PASSWORD")
        password = quote_plus(raw_password)
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        db = os.getenv("DB_NAME")

        # Credentials of mySQL connection
        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
        engine = create_engine(connection_string)
        print("Connection created Successfully")
        return engine
    except Exception as e:
        print(f"Error creating connection with DB: {e}")
        return None
    
# Initialize only once
engine = create_connection()
if not engine:
    print("Database connection failed.")
    exit()

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
def create_rag_chain(doc, prompt, parser, score_threshold=1.0):
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

    main_chain = parallel_chain | prompt | llm | parser
    return main_chain

# Function to Schedule Interview
def schedule_interview(role:str|dict, resume_path:str, question_limit:int, sender_email:str) -> str:
    # Skills and Experience fetching prompt and JSON Parser
    resume_parser = JsonOutputParser()

    # For Current Day
    current_month_year = datetime.now().strftime("%B %Y")

    resume_prompt = PromptTemplate(
        template="""
    You are an AI Resume Analyzer. Analyze the resume text below and extract **only** information relevant to the given job role.

    Your output **must** be in the following JSON format:
    {format_instruction}

    **Instructions:**
    1. **Name**:
    - Extract the candidate's full name from the **first few lines** of the resume.
    - It is usually the **first large bold text** or line that is **not an address, email, or phone number**.
    - Exclude words like "Resume", "Curriculum Vitae", "AI", or job titles.
    - If the name appears to be broken across lines, reconstruct it (e.g., "Abhis" and "hek" should be "Abhishek").
    - If no clear name is found, return: `"Name": "NA"`.

    2. **Skills**:
    - Extract technical and soft skills relevant to the **target role**.
    - Exclude generic or irrelevant skills (e.g., MS Word, Internet Browsing).
    - If **no skills are relevant**, return an empty list: `"Skills": []`.

    3. **Experience**:
    - Calculate the **cumulative time spent at each company** to get total professional experience.
    - Include only non-overlapping, clearly dated experiences (internships, jobs).
    - If a role ends in "Present" or "Current", treat it as ending in **{current_month_year}**.
    - Example: 
        - Google: Jan 2023 - Mar 2023 = 2 months  
        - Jobma: Feb 2025 - May 2025 = 3 months  
        - Total: 5 months = `"Experience": "0.42 years"`
    - Round the final answer to **2 decimal places**.
    - If durations are missing or unclear, return: `"Experience": "NA"`.

    4. Fetch email id from the document
    - Extract the first valid email address ending with `@gmail.com` from the text.
    - If not found, return `"Email": "NA"`.

    5. **Phone**:
    - Extract the first 10-digit Indian mobile number (starting with 6-9) from the resume.
    - You can allow formats with or without `+91`, spaces, or dashes.
    - Examples: `9876543210`, `+91-9876543210`, `+91 98765 43210`.
    - If no valid number is found, return `"Phone": "NA"`.

    6. **Education**:
    - Extract **highest qualification** (e.g., B.Tech, M.Tech, MCA, MBA, PhD).
    - Include the **degree name**, **specialization** (if available), and **university/institute name**.
    - Example: `"Education": "MCA in Computer Applications from VIPS, GGSIPU"`
    - If not found, return `"Education": "NA"`.
    ---

    **Target Role**: {role}

    **Resume Text**:
    {context}
    """,
        input_variables=["context", "role"],
        partial_variables={
            "format_instruction": resume_parser.get_format_instructions(),
            "current_month_year": current_month_year
        }
    )

    if not isinstance(resume_path, str):
        raise ValueError("resume_path must be a valid string")
    
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume file not found at path: {resume_path}")
    
    # Create the chain with JsonOutputParser instead of StrOutputParser
    resume_chain = resume_prompt | llm | resume_parser
    resume_result = resume_chain.invoke({'context': extract_document(resume_path), 'role': role})

    name = resume_result.get("Name", "NA")
    email = resume_result.get("Email", "NA")
    experience = resume_result.get("Experience", "NA")
    skills = ", ".join(resume_result.get("Skills", []))
    education = resume_result.get("Education", "NA")
    phone = resume_result.get("Phone", "NA")
    current_time = datetime.now()

    with engine.begin() as conn:  # Ensures transactional safety: commits on success, rolls back on error.
        # 1. Check if candidate exists
        result = conn.execute(text(
            "Select id from AI_INTERVIEW_PLATFORM.candidates Where email = :email"
        ),
        {"email": email}
        ).fetchone()

        if result:
            candidate_id = result[0]
        else:
            # 2. Insert candidate
            insert_candidate = text("""
                Insert into AI_INTERVIEW_PLATFORM.candidates (name, email, skills, education, experience, resume_path, phone, created_at)
                Values (:name, :email, :skills, :education, :experience, :resume_path, :phone, :created_at)
            """)
            conn.execute(insert_candidate, {
                "name": name,   
                "email": email,
                "skills": skills,
                "education": education,
                "experience": experience,
                "resume_path": resume_path,
                "phone": phone,
                "created_at": current_time
            })

            # Get new candidate_id
                # Scalar fetches the first column of the first row of the result set, or returns None if there are no rows.
            candidate_id = conn.execute(
                text("SELECT id FROM AI_INTERVIEW_PLATFORM.candidates WHERE email = :email"),
                {"email": email}
            ).scalar()

        # 3. Insert interview_invitation
        insert_invite = text("""
            INSERT INTO AI_INTERVIEW_PLATFORM.interview_invitation 
                (candidate_id, role, question_limit, sender_email, status, created_at, interview_scheduling_time)
            VALUES 
                (:candidate_id, :role, :question_limit, :sender_email, :status, :created_at, :interview_scheduling_time)
            """)
        
        conn.execute(insert_invite, {
            "candidate_id": candidate_id,
            "role": role,
            "question_limit": question_limit,
            "sender_email": sender_email,
            "status": "Scheduled",
            "created_at": current_time,
            "interview_scheduling_time": current_time
        })

    return f"Interview scheduled for '{name}' for role: {role}"

# Function to Track Candidate's Details
def track_candidate(filter: TrackCandidateInput) -> Union[list[dict], str]: 
    """Flexible candidate tracker. Filter by name, email, role, date, and interview status."""
    try:
        query = """
            SELECT 
                c.id AS candidate_id,
                c.name AS name,
                c.email AS email,
                c.phone AS phone,

                t.role AS role,
                t.sender_email AS sender_email,
                t.status AS status,
                t.interview_scheduling_time AS interview_scheduling_time,

                d.achieved_score AS achieved_score,
                d.total_score AS total_score,
                d.summary AS summary,
                d.recommendation AS recommendation,
                d.skills AS skills

            FROM AI_INTERVIEW_PLATFORM.candidates c
            LEFT JOIN AI_INTERVIEW_PLATFORM.interview_invitation t ON c.id = t.candidate_id
            LEFT JOIN AI_INTERVIEW_PLATFORM.interview_details d ON t.id = d.invitation_id
            WHERE 1=1
        """
        params = {}

        if filter.name:
            query += " AND LOWER(c.name) LIKE :name"
            params["name"] = f"%{filter.name.strip().lower()}%"

        if filter.email:
                query += " AND c.email = :email"
                params["email"] = filter.email.strip().lower()

        if filter.role:
            query += " AND LOWER(t.role) LIKE :role"
            params["role"] = f"%{filter.role.lower()}%" 

        if filter.status:
            query += " AND LOWER(t.status) = :status"
            params["status"] = filter.status.lower()

        if filter.date_filter:
            today = datetime.today()
            if filter.date_filter == "last_week":
                start = today - timedelta(days=today.weekday() + 7)
                end = start + timedelta(days=6)
            elif filter.date_filter == "recent":
                start = today - timedelta(days=3)
                end = today
            elif filter.date_filter == "today":
                start = today.replace(hour=0, minute=0, second=0, microsecond=0)
                end = today
            else:
                start = None

            if start:
                query += " AND t.interview_scheduling_time BETWEEN :start AND :end"
                params["start"] = start
                params["end"] = end
        
        query += " ORDER BY c.created_at DESC"

        with engine.begin() as conn:
            result = conn.execute(text(query), params).mappings().all()

        if not result:
            return "No matching candidate records found."
        return [dict(row) for row in result]
    
    except Exception as e:
        return f"Error in tracking candidates: {str(e)}"
    
# To check available roles
def list_all_scheduled_roles() -> Union[list[str], str]:
    """Returns a list of all distinct roles for which interviews are scheduled."""
    try:
        query = """
            SELECT DISTINCT role from AI_INTERVIEW_PLATFORM.interview_invitation
            WHERE role is not NULL
            Order by role
        """
        with engine.begin() as conn:
            result = conn.execute(text(query)).scalars().all()

        if not result:
            return "No roles found with scheduled interviews."
        return result
    except Exception as e:
        return f"Error fetching roles: {str(e)}"

# Parsing part of Track Candidate
def extract_filters(user_input:str) -> dict:
    # Parsing Prompt
    parsing_prompt = PromptTemplate(
        template="""
    You are a helpful assistant that extracts filters to track a candidate's interview information.
    Based on the user's request, extract and return a JSON object with the following keys:

    - name: Candidate's name (if mentioned, like "Priya Sharma", "Dushyant Goyal")
    - email: Candidate's email (e.g., "abc@example.com", "SinghDeepanshu1233@gmail.com")
    - role: Role mentioned (like "backend", "frontend", "data analyst", "AI associate", etc.)
    - date_filter: One of: "today", "recent", "last_week", or null if not mentioned
    - status: "Scheduled" or "Completed" if mentioned (e.g., "show scheduled interviews" → "Scheduled")

    Special cases:
    - If user asks for "scheduled" or "upcoming" interviews, set status to "Scheduled"
    - If user asks for "completed" or "past" interviews, set status to "Completed"

    Only include relevant values. If a value is not mentioned, return null.

    Input: {input}
    Output:
    """,
        input_variables=["input"]
    )

    parsing_chain = parsing_prompt | llm | JsonOutputParser()
    parsing_result = parsing_chain.invoke({"input": user_input})

    return parsing_result

# Tools   
interview_tool = StructuredTool.from_function(
    func=schedule_interview,
    name='schedule_interview',
    description="Extracts resume information and schedules interview. Input should be a dictionary with keys: role, resume_path, question_limit, sender_email",
    args_schema=ScheduleInterviewInput
)   

track_candidate_tool = StructuredTool.from_function(
    func=track_candidate,
    name='track_candidate',
    description="Track candidates. Provide email to get specific candidate details or leave blank to get a summary of all interviews.",
    args_schema=TrackCandidateInput
)

status_tool = StructuredTool.from_function(
    func=list_all_scheduled_roles,
    name="list_all_scheduled_roles",
    description="Returns a list of all distinct roles for which interviews are scheduled."
)

tools = [interview_tool, track_candidate_tool, status_tool]

# Agent
memory = ConversationBufferMemory(k=20, memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=3
)

# LangGraph =================================================================================================================>

# State Definition
class State(TypedDict):   
    question: str
    category: Literal["math", "irrelevant", "help", "greet", "bye", "agent"]
    answer: str
    history: List[Tuple[str,str]]
    role: str   
    resume_path: str
    question_limit: int
    sender_email: EmailStr
    result: str


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
    - **irrelevant**: The user input is unrelated to the Jobma platform or job interviews, such as asking about food, weather, sports, or general unrelated queries (e.g., "I want to make a pizza").
    - **math**: The user is asking a math-related question or calculation (e.g., "What is 3 + 5?", "Calculate 10/2", "Solve 2*3+1").
    - **agent**: The user is asking about scheduling an interview, tracking candidates, or any hiring-related tasks that involve actions like storing or retrieving data.
    
    Classify the following user input strictly as one of the intents above. Your response must be a **single word** from the list: `greet`, `help`, `bye`, `math`, or `irrelevant`.

    User Input:
    "{input}"

    Intent:
    """,
        input_variables=['input']
    )
    formatted_prompt = intent_prompt.format(input=question)
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    category = response.content.strip().lower()

    if category not in ['greet', 'help', 'bye', 'irrelevant', 'math', "agent"]:
        category = 'irrelevant'    # Fallback
    
    return {'category': category}

# Math Node
def math_node(state: State) -> dict:
    question = state.get("question", "")
    history = state.get("history", [])
    response = llm.invoke([HumanMessage(content=question)])
    answer = response.content
    history.append((question, answer))
    return {"answer": answer, "history": history}

# RAG Node
def rag_node(state: State) -> dict:
    question = state.get("question", "")
    history = state.get("history", [])
    rag_chain = create_rag_chain("Sensitive_Documents/formatted_QA.txt", rag_prompt, parser)
    answer = rag_chain.invoke(question)
    return {"answer": answer, "history": history}

# General LLM Node
def irrelevant_node(state:State) -> dict:
    return {"answer": "Sorry! I can only answer Jobma-Related Questions"}

# Greet Node
def greet_node(state: State) -> dict:
    question = state.get("question", "")
    response = llm.invoke([HumanMessage(content=question)])
    return {"answer": response.content}

# Bye Node
def bye_node(state: State) -> dict:
    return {"answer": "Goodbye! \nTake care and feel free to return anytime you need help."}

# Agent Node
def agent_node(state: State) -> dict:
    question = state.get('question', '')
    history = state.get('history', [])

    try:
        # Run the Agent
        agent_response = agent.run(question)

        # Update state with response
        return {
            **state,
            "answer" : agent_response,
            "history": history + [(question, agent_response)]
        }
    
    except Exception as e:
        error_message = f"Agent failed: {str(e)}"
        return {
            **state,
            "answer": error_message,
            "history": history + [(question, error_message)]
        }


# Build the LangGraph
builder = StateGraph(State)  # Main Graph

builder.add_node("router", router_node)
builder.add_node("rag_node", rag_node)
builder.add_node("math_node", math_node)
builder.add_node("irrelevant_node", irrelevant_node)
builder.add_node("greet_node", greet_node)
builder.add_node("bye_node", bye_node)
builder.add_node("agent_node", agent_node)

# Edges
builder.set_entry_point("router")
builder.add_conditional_edges(
    "router",
    lambda state: state['category'],
    {
        "math": "math_node",
        "irrelevant": "irrelevant_node",
        "help": "rag_node",
        "greet": "greet_node",
        "bye": "bye_node",
        "agent": "agent_node"
    }
)

builder.add_edge("math_node", END)
builder.add_edge("irrelevant_node", END)
builder.add_edge("rag_node", END)
builder.add_edge("greet_node", END)
builder.add_edge("bye_node", END)
builder.add_edge("agent_node", END)

# Compile the App
graph = builder.compile()

# Visualizing the Graph
print("\nGraph Structure:")
print(graph.get_graph().print_ascii())
print("=======================================================================================")

def ask_ai():
    result = {"history": []}
    while(True):
        user_question = input("Ask Something: ")
        
        if user_question == 'exit':
            break

        result = graph.invoke({
            "question": user_question,
            "history": result.get("history", []) if "history" in result else []
        })
        print("Answer: ", result['answer'])
    
    print("\n Full Conversation:")
    for q, a in result.get("history", []):
        print(f"Human: {q}")
        print(f"AI: {a}\n")

# Run the App
if __name__ == "__main__":
    ask_ai()