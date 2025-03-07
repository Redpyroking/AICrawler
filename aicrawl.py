import os
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Tavily search API
from langchain_community.tools.tavily_search import TavilySearchResults

# Web crawling tools
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector DB for storing processed information
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize LLM models for different agents
research_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
drafter_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)
qa_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)

# Initialize Tavily API
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(max_results=5)

# Initialize web crawler
google_search = GoogleSearchAPIWrapper()

# Initialize vector database with OpenAI embeddings
embeddings = OpenAIEmbeddings()
vector_db = Chroma(embedding_function=embeddings)

# Initialize text splitter for processing web content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# State type definition for the graph
class ResearchState(dict):
    """State for the research graph."""
    
    def __init__(
        self,
        query: str = "",
        expanded_queries: List[str] = None,
        search_results: List[Dict] = None,
        crawled_content: List[Dict] = None,
        processed_information: List[Dict] = None,
        draft_answer: str = "",
        final_answer: str = "",
        errors: List[str] = None,
    ):
        self.query = query
        self.expanded_queries = expanded_queries or []
        self.search_results = search_results or []
        self.crawled_content = crawled_content or []
        self.processed_information = processed_information or []
        self.draft_answer = draft_answer
        self.final_answer = final_answer
        self.errors = errors or []
        
        super().__init__(
            query=self.query,
            expanded_queries=self.expanded_queries,
            search_results=self.search_results,
            crawled_content=self.crawled_content,
            processed_information=self.processed_information,
            draft_answer=self.draft_answer,
            final_answer=self.final_answer,
            errors=self.errors,
        )

# 1. Query Expansion Agent
def query_expansion_agent(state: ResearchState) -> ResearchState:
    """Expands the original query into sub-queries for deeper research."""
    
    prompt = ChatPromptTemplate.from_template("""
    You are a query expansion agent. Your job is to take a complex research query and break it down 
    into 3-5 specific sub-queries that will help gather comprehensive information on the topic.
    
    Original Query: {query}
    
    Generate 3-5 specific sub-queries that together will provide comprehensive coverage of the topic.
    Format your response as a JSON list of strings.
    """)
    
    chain = prompt | research_llm | JsonOutputParser()
    
    try:
        expanded_queries = chain.invoke({"query": state["query"]})
        state["expanded_queries"] = expanded_queries
    except Exception as e:
        state["errors"].append(f"Query expansion error: {str(e)}")
        # Fallback to original query if expansion fails
        state["expanded_queries"] = [state["query"]]
    
    return state

# 2. Search Agent using Tavily
def search_agent(state: ResearchState) -> ResearchState:
    """Performs searches using Tavily API for each expanded query."""
    
    all_results = []
    
    for query in state["expanded_queries"]:
        try:
            results = tavily_tool.invoke({"query": query})
            # Add the source query to each result
            for result in results:
                result["source_query"] = query
            all_results.extend(results)
        except Exception as e:
            state["errors"].append(f"Search error for query '{query}': {str(e)}")
    
    # Sort results by relevance if available
    if all_results and "relevance_score" in all_results[0]:
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    state["search_results"] = all_results
    return state

# 3. Web Crawling Agent
def web_crawling_agent(state: ResearchState) -> ResearchState:
    """Crawls the URLs from search results to extract detailed content."""
    
    all_crawled_content = []
    
    # Get unique URLs from search results
    urls = [result["url"] for result in state["search_results"]]
    unique_urls = list(set(urls))
    
    # Limit to top 10 URLs to avoid too much processing
    for url in unique_urls[:10]:
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(documents)
            
            # Store the chunks with metadata
            for chunk in chunks:
                all_crawled_content.append({
                    "url": url,
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                })
                
        except Exception as e:
            state["errors"].append(f"Crawling error for URL '{url}': {str(e)}")
    
    state["crawled_content"] = all_crawled_content
    return state

# 4. Content Processing Agent
def content_processing_agent(state: ResearchState) -> ResearchState:
    """Processes crawled content to extract relevant information."""
    
    prompt = ChatPromptTemplate.from_template("""
    You are a content processing agent. Your job is to analyze the provided web content 
    and extract the most relevant information related to the original query.
    
    Original Query: {query}
    
    Web Content:
    {content}
    
    Extract the most relevant information, summarize it clearly, and include any key facts, 
    figures, or quotes. Format as markdown with clear sections.
    """)
    
    chain = prompt | research_llm | StrOutputParser()
    
    processed_information = []
    
    # Process each content chunk
    for item in state["crawled_content"]:
        try:
            processed = chain.invoke({
                "query": state["query"],
                "content": item["content"]
            })
            
            processed_information.append({
                "url": item["url"],
                "processed_content": processed,
                "metadata": item["metadata"]
            })
            
            # Add to vector database for retrieval
            vector_db.add_texts(
                texts=[processed],
                metadatas=[{"url": item["url"], "source_query": state["query"]}]
            )
            
        except Exception as e:
            state["errors"].append(f"Processing error: {str(e)}")
    
    state["processed_information"] = processed_information
    return state

# 5. Answer Drafting Agent
def answer_drafting_agent(state: ResearchState) -> ResearchState:
    """Drafts a comprehensive answer based on processed information."""
    
    # Retrieve the most relevant processed information from vector DB
    results = vector_db.similarity_search(state["query"], k=10)
    
    # Combine all relevant information
    all_info = "\n\n".join([doc.page_content for doc in results])
    
    prompt = ChatPromptTemplate.from_template("""
    You are an answer drafting agent. Your job is to create a comprehensive, well-structured 
    response to the original query based on the research information provided.
    
    Original Query: {query}
    
    Research Information:
    {information}
    
    Draft a comprehensive answer that addresses the query fully. Use a clear structure with 
    headings and subheadings. Include relevant facts, figures, and insights from the research.
    Format your response using markdown for readability.
    """)
    
    chain = prompt | drafter_llm | StrOutputParser()
    
    try:
        draft_answer = chain.invoke({
            "query": state["query"],
            "information": all_info
        })
        state["draft_answer"] = draft_answer
    except Exception as e:
        state["errors"].append(f"Draft generation error: {str(e)}")
        state["draft_answer"] = "Unable to generate draft due to an error."
    
    return state

# 6. Quality Assurance Agent
def quality_assurance_agent(state: ResearchState) -> ResearchState:
    """Reviews and improves the draft answer."""
    
    prompt = ChatPromptTemplate.from_template("""
    You are a quality assurance agent. Your job is to review and improve the draft answer 
    to ensure it's comprehensive, accurate, well-structured, and directly addresses the query.
    
    Original Query: {query}
    
    Draft Answer:
    {draft}
    
    Review the draft for:
    1. Comprehensiveness - Does it fully address all aspects of the query?
    2. Accuracy - Is all information factually correct?
    3. Structure - Is it well-organized with clear sections?
    4. Clarity - Is the writing clear and easy to understand?
    5. Citations - Are sources properly credited where needed?
    
    Provide an improved version of the answer, addressing any issues found.
    """)
    
    chain = prompt | qa_llm | StrOutputParser()
    
    try:
        final_answer = chain.invoke({
            "query": state["query"],
            "draft": state["draft_answer"]
        })
        state["final_answer"] = final_answer
    except Exception as e:
        state["errors"].append(f"QA error: {str(e)}")
        # Fallback to draft if QA fails
        state["final_answer"] = state["draft_answer"]
    
    return state

# 7. Error Handling Node
def error_handling_node(state: ResearchState) -> ResearchState:
    """Handles any errors that occurred during the process."""
    
    if state["errors"]:
        prompt = ChatPromptTemplate.from_template("""
        You are an error recovery agent. The research process encountered some errors.
        
        Original Query: {query}
        
        Errors encountered:
        {errors}
        
        Current draft answer:
        {draft}
        
        Please update the answer to acknowledge any limitations due to these errors and
        provide the best possible response with the information that was successfully gathered.
        """)
        
        chain = prompt | qa_llm | StrOutputParser()
        
        try:
            recovered_answer = chain.invoke({
                "query": state["query"],
                "errors": "\n".join(state["errors"]),
                "draft": state["final_answer"] or state["draft_answer"] or "No draft available."
            })
            state["final_answer"] = recovered_answer
        except Exception as e:
            # At this point, just return what we have with an error note
            if state["final_answer"]:
                state["final_answer"] += "\n\nNote: Some errors occurred during research."
            else:
                state["final_answer"] = "Unable to complete research due to technical errors."
    
    return state

# Define the agent workflow graph
def create_research_graph():
    """Creates the LangGraph for the research process."""
    
    workflow = StateGraph(ResearchState)
    
    # Add all nodes
    workflow.add_node("query_expansion", query_expansion_agent)
    workflow.add_node("search", search_agent)
    workflow.add_node("web_crawling", web_crawling_agent)
    workflow.add_node("content_processing", content_processing_agent)
    workflow.add_node("answer_drafting", answer_drafting_agent)
    workflow.add_node("quality_assurance", quality_assurance_agent)
    workflow.add_node("error_handling", error_handling_node)
    
    # Define the edges (workflow)
    workflow.add_edge("query_expansion", "search")
    workflow.add_edge("search", "web_crawling")
    workflow.add_edge("web_crawling", "content_processing")
    workflow.add_edge("content_processing", "answer_drafting")
    workflow.add_edge("answer_drafting", "quality_assurance")
    workflow.add_edge("quality_assurance", "error_handling")
    workflow.add_edge("error_handling", END)
    
    # Set the entry point
    workflow.set_entry_point("query_expansion")
    
    return workflow.compile()

# Main function to run the research system
def run_deep_research(query: str) -> str:
    """Runs the deep research system with the given query."""
    
    # Initialize the research graph
    research_graph = create_research_graph()
    
    # Initialize the state with the query
    initial_state = ResearchState(query=query)
    
    # Run the graph
    final_state = research_graph.invoke(initial_state)
    
    # Return the final answer
    return final_state["final_answer"]

# Example usage
if __name__ == "__main__":
    query = "What are the latest developments in quantum computing and its potential impact on cryptography?"
    result = run_deep_research(query)
    print(result)