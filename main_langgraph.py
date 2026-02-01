import os
from bs4 import BeautifulSoup
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from playwright.sync_api import sync_playwright

# Load environment variables
load_dotenv()

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

# Node 1: Scraper (Upgraded to Browser-based)
def scraper_node(state: AgentState):
    combined_text = ""
    
    with sync_playwright() as p:
        # Note: Added slow_mo to help with dynamic rendering
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        for url in state['urls']:
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
                
                # Scroll to bottom to trigger lazy loading
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(2000) 
                
                # Extract clean text from the whole page
                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove irrelevant tags to save tokens
                for tag in soup(["script", "style", "nav", "footer", "header", "svg"]):
                    tag.extract()
                    
                text = soup.get_text(separator=' ')
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = '\n'.join(chunk for chunk in chunks if chunk)
                
                combined_text += f"\n--- Content from {url} ---\n{clean_text}\n"
                
                # Specifically capture all interactive text
                interactives = page.query_selector_all("button, a, input[type='submit'], [role='button']")
                elements_found = []
                for el in interactives:
                    txt = el.inner_text().strip()
                    if txt:
                        elements_found.append(txt)
                
                if elements_found:
                    combined_text += f"\nInteractive elements found on {url}: " + ", ".join(list(set(elements_found))) + "\n"
                    
            except Exception as e:
                combined_text += f"\n--- Error scraping {url}: {str(e)} ---\n"
        
        browser.close()
    
    return {"scraped_content": combined_text}

# Node 2: Analyst
def analyst_node(state: AgentState):
    query = state['query']
    content = state['scraped_content']
    
    system_prompt = SystemMessage(content="""You are a precise data extraction specialist. 
    Your goal is to extract EXACT steps, button labels, and navigation paths from the scraped content.
    If you see buttons like "Add Attendance", "Submit", "Select Subject", report them specifically.
    Look for patterns that indicate a process (e.g., "Step 1", "Click here").
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
    Cite the specific button names and page URLs found in the scrape.
    Do not include internal logs or metadata in your response. Just the guide.
    If information is missing, politely explain what you saw and what the user might try.""")
    
    user_prompt = HumanMessage(content=f"Query: {query}\nAnalysis:\n{analysis}")
    
    response = llm.invoke([system_prompt, user_prompt])
    return {"final_response": response.content}

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("scrape", scraper_node)
workflow.add_node("analyze", analyst_node)
workflow.add_node("respond", responder_node)

# Add edges
workflow.set_entry_point("scrape")
workflow.add_edge("scrape", "analyze")
workflow.add_edge("analyze", "respond")
workflow.add_edge("respond", END)

# Compile
app = workflow.compile()

def run_assistant(user_query: str):
    urls = [
        "https://attendance-management-system-fronte-two.vercel.app/teacher/dashboard",
        "https://attendance-management-system-fronte-two.vercel.app/teacher/subject",
        "https://attendance-management-system-fronte-two.vercel.app/teacher/attendance"
    ]
    
    initial_state = {
        "query": user_query,
        "urls": urls,
        "scraped_content": "",
        "analysis": "",
        "final_response": ""
    }
    
    # Execute the graph
    result = app.invoke(initial_state)
    return result['final_response']

if __name__ == "__main__":
    query = input("How can I help you today? ")
    if query:
        response = run_assistant(query)
        print(f"\n{response}")
