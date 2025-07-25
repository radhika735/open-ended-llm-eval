import bm25s
import os
import logging
import json
from openai import OpenAI
from dotenv import load_dotenv
import Stemmer
from pydantic import BaseModel, Field
from typing import Annotated


load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)
# disable bm25s logging
logging.getLogger("bm25s").setLevel(logging.WARNING)

SYNOPSIS = "Bat Conservation"




def get_parsed_actions(synopsis=SYNOPSIS):
    """
    Get parsed actions from the data directory.
    
    Returns:
        list: List of parsed action dictionaries
    """
    parsed_actions = []
    data_dir = "action_data/key_messages/km_all"

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
                file_contents = file.read().strip()
                if file_contents:
                    lines = file_contents.splitlines()
                    # remove the line "Synopsis Details:" and lines after it
                    for i, line in enumerate(lines):
                        if line == "Synopsis Details:":
                            current_synopsis = lines[i+1]
                            lines = lines[:i]
                            break
                    # filter actions to only include the ones from the relevant synopsis
                    if current_synopsis == synopsis:
                        action_id, action_title = lines[0].split(": ", 1)
                        effectiveness = lines[1] if len(lines) > 1 else ""
                        key_messages = "\n".join(lines[2:]) if len(lines) > 2 else ""
                        parsed_actions.append({
                            "filename": filename,
                            "action_id": action_id.strip(),
                            "action_title": action_title.strip(),
                            "effectiveness": effectiveness.strip(),
                            "key_messages": key_messages.strip()
                        })
    
    return parsed_actions


def search_actions(query_string, k=3, offset=0):
    """
    Search for the top k most relevant actions based on a query string.
    
    Args:
        query_string (str): The search query for finding relevant actions
        k (int): Number of top results to return (default: 3)
        offset (int): Number of results to skip (default: 0, used for pagination)
    
    Returns:
        list: Top k action documents matching the query, starting from offset
    """
    parsed_actions = get_parsed_actions()
    
    corpus = []
    for action in parsed_actions:
        corpus.append(f"{action['action_id']}: {action['action_title']}\nEffectiveness: {action['effectiveness']}\nKey Messages:\n{action['key_messages']}")

    stemmer = Stemmer.Stemmer("english")

    # Tokenize the corpus and index it
    logging.debug("Tokenizing and indexing the corpus...")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    retriever = bm25s.BM25()
    logging.debug("Creating BM25 retriever...")
    retriever.index(corpus_tokens)

    logging.debug("BM25 retriever is ready.")
    
    # Search the corpus with the provided query
    # Retrieve more results than needed to handle offset
    total_results_needed = k + offset
    query_tokens = bm25s.tokenize(query_string)
    logging.debug(f"Searching for query: {query_string}")
    docs, scores = retriever.retrieve(query_tokens, k=total_results_needed)
    
    logging.debug(f"Docs: {docs}\nScores: {scores} (type: {type(scores)})")

    # Format results with metadata, applying offset and limit
    results = []
    for i, (doc, score) in enumerate(zip(docs[0], scores[0])):
        # Skip results before offset
        if i < offset:
            continue
        # Stop if we've collected enough results
        if len(results) >= k:
            break
            
        score = score.item()
        logging.debug(f"doc: {doc}")
        logging.debug(f"score: {score}, type: {type(score)}")
        action = parsed_actions[doc]
        results.append({
            "action_id": action["action_id"],
            "action_title": action["action_title"],
            "effectiveness": action["effectiveness"],
            "rank": i + 1  # Keep original rank from search
        })
    
    logging.debug(f"Found {len(results)} documents for query '{query_string}':")
    return results


def get_action_details(action_id):
    """
    Retrieve the full details for a specific action by its ID.
    
    Args:
        action_id (str): The action ID to retrieve (e.g., "1", "101", "1002")
    
    Returns:
        dict: Full action details or None if not found
    """
    parsed_actions = get_parsed_actions()
    
    # Find the action with matching ID
    for action in parsed_actions:
        if action["action_id"] == action_id:
            logging.debug(f"Found action details for ID: {action_id}")
            return {
                "action_id": action["action_id"],
                "action_title": action["action_title"],
                "effectiveness": action["effectiveness"],
                "key_messages": action["key_messages"],
                "filename": action["filename"]
            }
    
    logging.debug(f"Action ID {action_id} not found")
    return {
        "error": f"Action with ID '{action_id}' not found",
        "available_ids": [action["action_id"] for action in parsed_actions[:10]]  # Show first 10 as examples
    }




# Tool definition for OpenAI/OpenRouter function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_actions",
            "description": "Search for the top k most relevant action documents based on a query string. Returns action documents with their relevance scores and metadata. Use offset for pagination to retrieve additional results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_string": {
                        "type": "string",
                        "description": "The search query for finding relevant actions (e.g., 'bee conservation', 'renewable energy', 'urban planning')"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of top results to return (default: 3, max: 10)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of results to skip for pagination (default: 0). Use this to retrieve additional results beyond the initial set.",
                        "default": 0,
                        "minimum": 0
                    }
                },
                "required": ["query_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_action_details",
            "description": "Retrieve the full details for a specific action by its ID. Use this after finding relevant actions with search_actions to get complete information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "description": "The action ID to retrieve details for (e.g., '1', '101', '1002')"
                    }
                },
                "required": ["action_id"]
            }
        }
    }
]

# Tool mapping for function execution
TOOL_MAPPING = {
    "search_actions": search_actions,
    "get_action_details": get_action_details
}


## Defining llm response format

class QueryAnswer(BaseModel):
    action_ids : Annotated[list[str], Field(description="List of IDs of all the actions used to generate the answer")]
    final_answer_to_query : Annotated[str, Field(description="Answer to the user's query")]

schema = QueryAnswer.model_json_schema()
schema.update({"additionalProperties" : False})

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_answer",
        "strict": True,
        "schema": schema
    }
}




# API Configuration
def get_client():
    """
    Get an OpenAI client configured for openrouter.
    
    Returns:
        OpenAI: Configured client
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def call_llm(messages, model):
    """
    Make a call to the LLM with tool capabilities.
    
    Args:
        messages (list): List of message objects
        model (str): Model to use for the completion
    
    Returns:
        The response from the LLM
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        tools=tools,
        response_format=response_format,
        messages=messages,
        extra_body={
            "require_parameters": True,
        }
    )
    
    # Add the assistant's response to messages
    messages.append(response.choices[0].message.model_dump())
    return response


def execute_tool_call(tool_call):
    """
    Execute a tool call and return the result.
    
    Args:
        tool_call: The tool call object from the LLM response
    
    Returns:
        dict: Tool response message for the conversation
    """
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    
    logging.info(f"Executing tool: {tool_name} with args: {tool_args}")
    
    # Execute the tool function
    tool_result = TOOL_MAPPING[tool_name](**tool_args)
    
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_name,
        "content": json.dumps(tool_result, indent=2),
    }


def run_agentic_loop(user_query, model="google/gemini-2.5-flash", max_iterations=10):
    """
    Run an agentic loop that can use tools to answer user queries.
    
    Args:
        user_query (str): The user's question or request
        model (str): Model to use for completions
        max_iterations (int): Maximum number of iterations to prevent infinite loops
    
    Returns:
        str: The final response from the assistant
    """
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that can search through action documents to find relevant information. 
            Use the search_actions tool to find relevant actions based on user queries, and then use get_action_details to retrieve full details for specific actions when needed. 
            When searching for actions, start with the default parameters, but if you need more results, use the offset parameter to retrieve additional actions. 
            Keep searching with increasing offset values until the returned actions become irrelevant to the user's query - this ensures you find all pertinent information before providing your final answer. 
            Cite the relevant action numbers you use in your final response.
            """
        },
        {
            "role": "user",
            "content": user_query,
        }
    ]
    
    logging.info(f"Starting conversation with query: {user_query}")
    
    iteration_count = 0
    
    while iteration_count < max_iterations:
        iteration_count += 1
        logging.debug(f"Iteration {iteration_count}: Calling LLM...")
        
        response = call_llm(messages, model)
        
        # Check if the model wants to use tools
        if response.choices[0].message.tool_calls:
            logging.debug("Processing tool calls...")
            logging.info(f"LLM requested {len(response.choices[0].message.tool_calls)} tool call(s)")
            
            # Execute all tool calls
            for tool_call in response.choices[0].message.tool_calls:
                tool_response = execute_tool_call(tool_call)
                messages.append(tool_response)
        else:
            # No more tool calls, we have the final response
            logging.info("Conversation complete.")
            logging.info(f"Final response: {response.choices[0].message.content}")
            break
    
    if iteration_count >= max_iterations:
        logging.warning("Warning: Maximum iterations reached")
    
        # Return the final assistant message
    final_message = messages[-1] if messages[-1]["role"] == "assistant" else messages[-2]
    return final_message.get("content", "No response generated")



if __name__ == "__main__":
    # Example usage
    query = "What are the most effective interventions for reducing bat fatalities at wind turbines?"
    response = run_agentic_loop(query)