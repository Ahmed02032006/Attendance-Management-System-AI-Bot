import os
import json
from bs4 import BeautifulSoup
from typing import TypedDict, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import httpx
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Define the state for the graph
class AgentState(TypedDict):
    query: str
    urls: List[str]
    scraped_content: str
    analysis: str
    final_response: str
    validation_status: str

# Node 1: Scraper (Vercel-friendly)
async def scraper_node(state: AgentState):
    combined_text = ""
    urls = state['urls']
    
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for url in urls:
            try:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Remove irrelevant tags
                    for tag in soup(["script", "style", "nav", "footer", "header", "svg"]):
                        tag.extract()
                        
                    text = soup.get_text(separator=' ')
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    clean_text = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    combined_text += f"\n--- Content from {url} ---\n{clean_text}\n"
                else:
                    combined_text += f"\n--- Failed to scrape {url} (Status: {response.status_code}) ---\n"
            except Exception as e:
                combined_text += f"\n--- Error scraping {url}: {str(e)} ---\n"
                
    return {"scraped_content": combined_text}

# Node 2: Analyst
def analyst_node(state: AgentState):
    query = state['query']
    content = state['scraped_content']
    
    system_prompt = SystemMessage(content="""You are a precise data extraction specialist. 
    Your goal is to extract EXACT steps, button labels, and navigation paths from the scraped content.
    If you see buttons like "Add Attendance", "Submit", "Select Subject", report them specifically.
    Look for patterns that indicate a process.
    Do not give general advice. Be specific to the labels found in the text.""")
    
    user_prompt = HumanMessage(content=f"Query: {query}\n\nScraped Content:\n{content}")
    
    response = llm.invoke([system_prompt, user_prompt])
    return {"analysis": response.content}

# Node 3: Responder
def responder_node(state: AgentState):
    query = state['query']
    analysis = state['analysis']
    
    system_prompt = SystemMessage(content="""You are an expert Technical Assistant for the Attendance System.
    Your tone must be helpful, direct, and conversational.
    Start your response with a phrase like "In order to [user query]..." or "To [user query], you should...".
    Format the response with clear headings and numbered lists.
    Do not include internal logs or metadata in your response. Just the guide.""")
    
    user_prompt = HumanMessage(content=f"Query: {query}\nAnalysis:\n{analysis}")
    
    response = llm.invoke([system_prompt, user_prompt])
    return {"final_response": response.content}

# Node 4: Validation Agent
def validation_node(state: AgentState):
    query = state['query']
    final_response = state['final_response']
    
    # First, classify the query type (complaint, feature request, bug, or normal question)
    system_prompt = SystemMessage(content="""You are a query classifier for a customer support system.
    Your job is to classify the user's query into one of these categories:
    
    Classify as "COMPLAINT_OR_ISSUE" if the query contains:
    - Complaints about system performance (slow, crashing, freezing, etc.)
    - Bug reports or error messages
    - Feature requests or suggestions for new features
    - Complaints about UI/UX (colors, design, usability)
    - Reports of broken functionality
    - Frustration with the system
    
    Classify as "NORMAL_QUESTION" if the query is:
    - A how-to question
    - Asking for instructions or guidance
    - Seeking information about existing features
    
    IMPORTANT: Focus ONLY on the user's query, NOT on the assistant's response.
    Even if the assistant provides a good answer to a complaint, it's still a complaint.
    
    Respond with ONLY one phrase: either "COMPLAINT_OR_ISSUE" or "NORMAL_QUESTION".""")
    
    user_prompt = HumanMessage(content=f"User Query: {query}")
    
    classification = llm.invoke([system_prompt, user_prompt])
    query_type = classification.content.strip().upper()
    
    # If it's a complaint or issue, log to Google Sheets
    if "COMPLAINT_OR_ISSUE" in query_type:
        try:
            # Get credentials from environment
            creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
            sheet_id = os.getenv("GOOGLE_SHEET_ID")
            
            if creds_json and sheet_id:
                # Parse credentials
                creds_dict = json.loads(creds_json)
                
                # Authenticate with Google Sheets
                scope = ['https://spreadsheets.google.com/feeds',
                         'https://www.googleapis.com/auth/drive']
                creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                client = gspread.authorize(creds)
                
                # Open the sheet and append only the query (not the response)
                sheet = client.open_by_key(sheet_id).sheet1
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sheet.append_row([timestamp, query])
                
                # Override the response with acknowledgment message
                acknowledgment = """Thank you for bringing this to our attention. We have recorded your feedback and our team will review this issue. We appreciate your patience and will work on resolving this as soon as possible.

If you have any urgent concerns, please contact our support team at: m.ahmedofficial677@gmail.com"""
                
                return {
                    "validation_status": "COMPLAINT/ISSUE - Logged to Google Sheets",
                    "final_response": acknowledgment
                }
            else:
                # No credentials, but still show acknowledgment
                acknowledgment = """Thank you for bringing this to our attention. We have recorded your feedback and our team will review this issue. We appreciate your patience and will work on resolving this as soon as possible.

If you have any urgent concerns, please contact our support team at: m.ahmedofficial677@gmail.com"""
                return {
                    "validation_status": "COMPLAINT/ISSUE - No Google Sheets credentials configured",
                    "final_response": acknowledgment
                }
        except Exception as e:
            # Error logging, but still show acknowledgment
            acknowledgment = """Thank you for bringing this to our attention. We have recorded your feedback and our team will review this issue. We appreciate your patience and will work on resolving this as soon as possible.

If you have any urgent concerns, please contact our support team at: m.ahmedofficial677@gmail.com"""
            return {
                "validation_status": f"COMPLAINT/ISSUE - Error logging: {str(e)}",
                "final_response": acknowledgment
            }
    
    return {"validation_status": "NORMAL_QUESTION - Not logged"}

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("scrape", scraper_node)
workflow.add_node("analyze", analyst_node)
workflow.add_node("respond", responder_node)
workflow.add_node("validate", validation_node)
workflow.set_entry_point("scrape")
workflow.add_edge("scrape", "analyze")
workflow.add_edge("analyze", "respond")
workflow.add_edge("respond", "validate")
workflow.add_edge("validate", END)
graph_app = workflow.compile()

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def query_attendance(request: QueryRequest):
    urls = [
        "https://attendance-management-system-fronte-two.vercel.app/teacher/dashboard",
        "https://attendance-management-system-fronte-two.vercel.app/teacher/subject",
        "https://attendance-management-system-fronte-two.vercel.app/teacher/attendance"
    ]
    
    initial_state = {
        "query": request.query,
        "urls": urls,
        "scraped_content": "",
        "analysis": "",
        "final_response": "",
        "validation_status": ""
    }
    
    try:
        # LangGraph invoke is used for the workflow
        result = await graph_app.ainvoke(initial_state)
        return {"response": result['final_response']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Attendance System Assistant API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

