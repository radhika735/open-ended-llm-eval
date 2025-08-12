import bm25s
import os
import logging
import json
from openai import OpenAI
import Stemmer
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated


### ORDER OF ITERATIONS OF EXPERIMENTS:
# ENCOURAGING PARALLEL TOOL CALLS (worked well so KEEPING this)
# DIFF ANSWER STYLES + PARALLEL CALLS


load_dotenv()

logging.basicConfig(filename = "logfiles/answer_gen_experimental.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)
# disable bm25s logging
logging.getLogger("bm25s").setLevel(logging.WARNING)


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


def get_parsed_actions():
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
                            lines = lines[:i]
                            break
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
    query_tokens = bm25s.tokenize(query_string, stopwords="en", stemmer=stemmer)
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


def get_formatted_result(query, answer, action_ids):
    """
    Formats the user's query, answer and action ids as a valid JSON object string, to be presented to the user.
    
    Args:
        query (str): The query the user originally asked.
        answer (str): The answer generated to the user's query.
        action_ids (list[str]): A list containing all the action IDs used to generate the answer.
    
    Returns:
        str: Formatted result.
    """
    formatted_result = {
        "query":query,
        "answer":answer,
        "action_ids":action_ids
    }
    return formatted_result
    

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
    },
    {
        "type": "function",
        "function": {
            "name": "get_formatted_result",
            "description": "Format the answer and action IDs obtained into a valid JSON object string. Use this as the last step before presenting the answer to the user.",
            "parameters":{
                "type": "object",
                "properties": {
                    "query":{
                        "type": "string",
                        "description": "The user's query."
                    },
                    "answer": {
                        "type": "string",
                        "description": "The answer generated to the user's query."
                    },
                    "action_ids":{
                        "type": "array",
                        "items": {"type":"string"},
                        "description": "List of action IDs used to generate the answer to the query."
                    }
                },
                "required": ["answer","action_ids"]
            }
        }
    }
]

# Tool mapping for function execution
TOOL_MAPPING = {
    "search_actions": search_actions,
    "get_action_details": get_action_details,
    "get_formatted_result" : get_formatted_result
}


# class AnswerAndActionIDs(BaseModel):
#     answer : Annotated[str, Field(description="Answer to the user's query")]
#     action_ids : Annotated[list[str], Field(description="List of IDs of all the actions used to generate the answer")]

# schema = AnswerAndActionIDs.model_json_schema()
# schema["type"] = "object"
# schema["additionalProperties"] = False


def call_llm(messages, model, provider):
    """
    Make a call to the LLM with tool capabilities.
    
    Args:
        messages (list): List of message objects
        model (str): Model to use for the completion
    
    Returns:
        The response from the LLM
    """
    client = get_client()
    if provider is not None:
        response = client.chat.completions.create(
            model=model,
            tools=tools,
            messages=messages,
            # reasoning={"exclude":True},
            extra_body={
                "require_parameters": True,
                "provider": {
                    "order": [f"{provider}"], # Specify the single provider you want to pin
                    "allow_fallbacks": False     # Set fallback to None to prevent routing elsewhere
                }
            }
        )
    else:
        response = client.chat.completions.create(
            model=model,
            tools=tools,
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
        "content": json.dumps(tool_result, indent=2)
    }


def run_agentic_loop(user_query, model="google/gemini-2.5-flash", provider=None, max_iterations=10):
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
            "content": """You are a helpful assistant that can search through action documents to find relevant information. 
            Use the search_actions tool to find relevant actions based on user queries, and then use get_action_details to retrieve full details for specific actions when needed. 
            When searching for actions, start with the default parameters, but if you need more results, use the offset parameter to retrieve additional actions. 
            Keep searching with increasing offset values until the returned actions become irrelevant to the user's query - this ensures you find all pertinent information before providing your final answer. 
            Once you have identified a list or batch of action documents that you want to expand fully, you should make multiple **parallel** calls to the get_action_details tool — each call should contain one of the action IDs from the batch.
            Use the information you have gathered to answer the user's query. Include as much detail in your answer as possible (but DO NOT make up statements that do not exist in the action documents).
            You must cite the action IDs as references for statements you make in your answer.
            Finally, you MUST use the get_formatted_result tool. This will convert your the user's query, your generated answer and the list of the action IDs you used to generate your answer into a valid JSON format. 
            You MUST copy the exact outputted json string of the get_formatted_result tool as your final message."
        """ # has a typo: "This will convert *your the* user's query".
        },
        {
            "role": "user",
            "content": user_query,
        }
    ]
    
    logging.info(f"Starting conversation with query: {user_query}")
    logging.info(f"Model: {model}")
    
    iteration_count = 0
    
    all_tool_calls = []# list of dicts, each dict containing key "iteration_num" : int, and key "tools_called" : []. 
                        # The list value associated with "tools_called", is a list of dicts itself.
                        # These dicts contain the following keys: "function_name" : str, "args" : str, "return_val" : str.

    while iteration_count < max_iterations:
        iteration_count += 1

        iteration_tool_calls = {"iteration_num":iteration_count, "tools_called":[]}
        all_tool_calls.append(iteration_tool_calls)

        logging.debug(f"Iteration {iteration_count}: Calling LLM...")
        
        response = call_llm(messages, model, provider)
        logging.info(f"Provider: {response.provider}")
        
        # Check if the model wants to use tools
        if response.choices[0].message.tool_calls:
            logging.debug("Processing tool calls...")
            logging.info(f"LLM requested {len(response.choices[0].message.tool_calls)} tool call(s)")
            
            # Execute all tool calls
            for tool_call in response.choices[0].message.tool_calls:
                tool_response = execute_tool_call(tool_call)

                single_tool_call_info = {"function_name":tool_call.function.name, "args":tool_call.function.arguments, "return_val":tool_response["content"]}
                iteration_tool_calls["tools_called"].append(single_tool_call_info)

                if tool_call.function.name == "get_formatted_result":
                    # The last execution of the tool call gave us the formatted final response 
                    result = json.loads(tool_response["content"])
                    logging.info("Conversation complete, with formatting tool call.")
                    logging.info(f"Final result: {result} ")
                    return result, all_tool_calls # SUCCESSFUL EXIT
                else:
                    messages.append(tool_response)
        # else:
        #     # No more tool calls, we have the final response
        #     logging.warning("Conversation complete, without formatting tool call.")
        #     logging.info(f"Final response: {response.choices[0].message.content}")
        #     break
    
    if iteration_count >= max_iterations:
        logging.warning("Warning: Maximum iterations reached")
    
        # Return the final assistant message
    final_message = messages[-1] if messages[-1]["role"] == "assistant" else messages[-2]
    final_message_content = final_message.get("content", "No response generated")
    logging.info(f"Final response (after hitting max iterations) {final_message_content}")
    return final_message_content, all_tool_calls# FAILED EXIT


def get_prev_answers(filename):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as file:
                ans_list = json.load(file)
                logging.info(f"Loaded existing answers from {filename}")
                success = True
        except json.JSONDecodeError as e:
            ans_list = []
            logging.warning(f"Failed to load existing answers from {filename}.")
            success = False
    else:
        ans_list = []
        logging.info(f"Creating new answer output file {filename}.")
        success = True

    if not isinstance(ans_list, list):
        raise ValueError("Expected JSON file to contain a list.")
    
    return success, ans_list


def assemble_llm_response_and_tools(response : dict, tool_use_track):
    # response is a dict with the properties 
    response["tool_calls"] = tool_use_track
    return response



def write_new_answers(ans_list, filename, reset_file = False):
    if not reset_file:
        success, all_answers = get_prev_answers(filename=filename)
        if not success:
            logging.warning(f"Could not load existing file content, so failed write to file {filename}.")
            return
        all_answers.extend(ans_list)
    else:
        all_answers = ans_list

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(all_answers, file, indent=2)
    logging.info(f"Updated {filename} with new questions.")


def parse_model_name(model):
    model_split = model.split("/")
    model_name = model_split[-1]
    cleaned_name = ""
    for char in model_name:
        if char.isalnum():
            cleaned_name += char
        else:
            cleaned_name += "-"
    return cleaned_name

def parse_provider_name(provider):
    if provider is not None:
        provider_split = provider.split("/")
        provider_name = provider_split[0]
        return provider_name
    else:
        return ""


def run_models(query, model_provider_list):
    for model, provider in model_provider_list:
        cleaned_model_name = parse_model_name(model)
        if provider is not None:
            cleaned_provider_name = parse_provider_name(provider)
            ans_out_file = f"answer_gen_data/experimental/parallel_tools/answers_{cleaned_provider_name}_{cleaned_model_name}.json"
        else:
            ans_out_file = f"answer_gen_data/experimental/parallel_tools/answers__{cleaned_model_name}.json"
        
        response, tool_calls = run_agentic_loop(user_query=query, model=model, provider=provider)
        write_new_answers(ans_list=[assemble_llm_response_and_tools(response, tool_calls)], filename=ans_out_file)


def run_queries_on_models(query_list, model_provider_list):
    for query in query_list:
        run_models(query=query, model_provider_list=model_provider_list)


def main():
    queries = [
        "How effective is providing artificial nesting sites for different types of bees?",
        "What are the documented trade-offs of using prescribed fire as a forest management tool?",
        "How can altering the type of food provided to captive primates improve their welfare?",
        "What are the most effective ways to modify man-made structures like windows and power lines to reduce bird collision mortality?",
        "When creating new habitats on farmland, how long does it typically take to see benefits for wildlife?"
    ]

    model_provider_list = [
        ("google/gemini-2.5-flash", None),
        ("moonshotai/kimi-k2", "fireworks/fp8"),
        ("qwen/qwen3-235b-a22b-thinking-2507", "together"),
        ("anthropic/claude-sonnet-4", None),
        ("anthropic/claude-opus-4", None),
        #("x-ai/grok-4", None),
        ("openai/gpt-4.1", None),
        ("google/gemini-2.5-pro", None),
        #("deepseek/deepseek-r1-0528", "novita/fp8"),
        #("mistralai/magistral-medium-2506", None),
        #("mistralai/magistral-medium-2506:thinking", None),
    ]
    # could also test gemini-2.5-flash-lite

    #query = "What conservation actions are most beneficial for establishing new populations of threatened toad species in the UK?"

    #run_models(query=query, model_provider_list=model_provider_list)

    # ## TESTING DIFF ANSWER STYLES
    # query = "What are the most effective interventions for reducing bat fatalities at wind turbines?"
    # model_name = "moonshotai/kimi-k2"
    # provider_name = "fireworks/fp8"
    # result, tool_calls = run_agentic_loop(user_query=query, model=model_name, provider=provider_name)
    # cleaned_model_name = parse_model_name(model_name)
    # cleaned_provider_name = parse_provider_name(provider_name)
    # # CHANGE ANSWER STYLE IN THE FILENAME
    # ans_out_file = f"answer_gen_data/experimental/diff_answer_styles/answers_{cleaned_provider_name}_{cleaned_model_name}_tldrpara.json"
    # write_new_answers(ans_list=[assemble_llm_response_and_tools(result, tool_calls)], filename=ans_out_file)

    ## TESTING SEARCH_ACTIONS()
    search_query = "chytridiomycosis"
    logging.info(f"TESTING search_actions for presence of action 762, 'Add salt to ponds to reduce chytridiomycosis', in relation to query about {search_query}.")
    results = search_actions(search_query)
    print(results)
    logging.info(f"Results: {results}.")
    logging.info("ENDING test.")


    ## TESTING GET_PARSED_ACTIONS()
    # search_query = "chytridiomycosis"
    # logging.info(f"TESTING get_parsed_actions for presence of action 762, 'Add salt to ponds to reduce chytridiomycosis', in relation to query about {search_query}.")
    # actions = get_parsed_actions()
    # ids = [doc["action_id"] for doc in actions]
    # titles = [doc["action_title"] for doc in actions]
    # if "762" in ids:
    #     logging.info("Action 762 found in parsed actions.")
    #     print("Action 762 found in parsed actions.")
    # else:
    #     logging.warning("Action 762 not found in parsed actions.")
    #     print("Action 762 not found in parsed actions.")
    # logging.info("ENDING test.")



if __name__ == "__main__":
    main()