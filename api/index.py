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
    
    # Use LLM to classify if the query was adequately answered
    system_prompt = SystemMessage(content="""You are a quality assurance agent.
    Your job is to determine if the user's query was adequately answered.
    
    Classify the query as:
    - "ANSWERED" if the response provides clear, actionable steps or information.
    - "UNANSWERED" if:
      * The query is a complaint about a feature
      * The query is a feature request
      * The response is vague or doesn't address the query
      * The user is reporting a bug or issue
    
    Respond with ONLY one word: either "ANSWERED" or "UNANSWERED".""")
    
    user_prompt = HumanMessage(content=f"User Query: {query}\n\nAssistant Response: {final_response}")
    
    classification = llm.invoke([system_prompt, user_prompt])
    status = classification.content.strip().upper()
    
    # If unanswered, log to Google Sheets
    if "UNANSWERED" in status:
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
                
                # Open the sheet and append the query
                sheet = client.open_by_key(sheet_id).sheet1
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sheet.append_row([timestamp, query, final_response])
                
                return {"validation_status": "UNANSWERED - Logged to Google Sheets"}
            else:
                return {"validation_status": "UNANSWERED - No Google Sheets credentials configured"}
        except Exception as e:
            return {"validation_status": f"UNANSWERED - Error logging: {str(e)}"}
    
    return {"validation_status": "ANSWERED"}

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
