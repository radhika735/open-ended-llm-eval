import time
import os
import logging
import json
from openai import OpenAI
from dotenv import load_dotenv
import copy

from utils.action_parsing import ActionParsingContext, get_parsed_action_by_id
from utils.rag import sparse_retrieve_docs, dense_retrieve_docs, hybrid_retrieve_docs
from utils.exceptions import RetrievalError, FatalAPIError


load_dotenv()


LLM_ACTION_CONTEXT = ActionParsingContext(
    required_fields=["action_id", "action_title", "key_messages"]
)
RAG_ACTION_CONTEXT = ActionParsingContext(
    required_fields=["action_id", "action_title", "key_messages", "background_information"]
)
RETRIEVAL_TYPE = "hybrid" # other options: "dense", "hybrid".
if RETRIEVAL_TYPE == "hybrid":
    FUSION_TYPE = "cross-encoder" # other option: "reciprocal rank fusion"


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


def search_actions(query_string, k=3, offset=0):
    """
    Search for the top k most relevant action documents based on a query string.
    """
    if RETRIEVAL_TYPE == "sparse":
        return sparse_retrieve_docs(query_string=query_string, context=RAG_ACTION_CONTEXT, k=k, offset=offset)
    elif RETRIEVAL_TYPE == "dense":
        return dense_retrieve_docs(query_string=query_string, context=RAG_ACTION_CONTEXT, k=k, offset=offset)
    elif RETRIEVAL_TYPE == "hybrid":
        if FUSION_TYPE not in ["cross-encoder", "reciprocal rank fusion"]:
            logging.warning("Invalid FUSION_TYPE set for hybrid retrieval of action documents. Must be either 'cross-encoder' or 'reciprocal rank fusion'. Defaulting to using cross-encoder.")
            fusion_type = "cross-encoder"
        else:
            fusion_type = FUSION_TYPE
        return hybrid_retrieve_docs(query_string=query_string, context=RAG_ACTION_CONTEXT, fusion_type=fusion_type, k=k, offset=offset)
    else:
        logging.warning("Invalid RETRIEVAL_TYPE set for retrieving action documents by similarity to query string. Must be either 'sparse', 'dense' or 'hybrid'. Defaulting to sparse retrieval.")
        return sparse_retrieve_docs(query_string=query_string, context=RAG_ACTION_CONTEXT, k=k, offset=offset)


def get_action_details(action_id):
    """
    Retrieve the full details for a specific action by its ID.
    
    Args:
        action_id (str): The action ID to retrieve (e.g., "1", "101", "1002")
    
    Returns:
        dict: Full action details or None if not found
    """
    parsed_action = get_parsed_action_by_id(id=action_id, context=LLM_ACTION_CONTEXT)
    if parsed_action is not None:
        logging.info(f"Found action details for ID: {action_id}")
        return parsed_action
    else:
        logging.warning(f"Action ID {action_id} not found.")
        example_ids = ["1000", "1001", "1002", "1003", "1005", "1006", "1007", "1008", "1009", "100"]
        return {
            "error": f"Action with ID '{action_id}' not found",
            "available_ids": example_ids # Show some (ten) action ids as examples.
        }


def get_formatted_result(query, summary, action_ids):
    """
    Formats the user's query, the compiled summary and action ids as a valid JSON object string, to be presented to the user.
    
    Args:
        query (str): The query the user originally asked.
        relevant_summary (str): The summary compiled from information relevant to the user's query.
        action_ids (list[str]): A list containing all the action IDs used to generate the summary.
    
    Returns:
        str: Formatted result.
    """
    formatted_result = {
        "query":query,
        "relevant_summary":summary,
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
            "description": "Format the summary and action IDs obtained into a valid JSON object string. Use this as the last step before presenting the summary to the user.",
            "parameters":{
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query."
                    },
                    "summary": {
                        "type": "string",
                        "description": "The generated summary of information relevant to the user's query."
                    },
                    "action_ids":{
                        "type": "array",
                        "items": {"type":"string"},
                        "description": "List of action IDs used to generate the summary."
                    }
                },
                "required": ["summary","action_ids"]
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


def call_llm(messages, model, provider):
    """
    Make a call to the LLM with tool capabilities.
    
    Args:
        messages (list): List of message objects
        model (str): Model to use for the completion
    
    Returns:
        The response from the LLM
    """
    try:
        client = get_client()
    except ValueError as e:
        raise FatalAPIError(f"Error initializing OpenAI client: {e}")
    
    try:
        if provider is not None:
            response = client.chat.completions.create(
                model=model,
                tools=tools,
                messages=messages,
                reasoning_effort="low",
                # reasoning={"exclude":True},
                extra_body={
                    "require_parameters": True,
                    "provider": {
                        "order": [f"{provider}"], # Specify the single provider you want to pin
                        "allow_fallbacks": False     # Set fallback to None to prevent routing elsewhere
                    },
                    "usage":{
                        "include":True
                    }
                }
            )
        else:
            response = client.chat.completions.create(
                model=model,
                tools=tools,
                messages=messages,
                reasoning_effort="low",
                extra_body={
                    "require_parameters": True,
                    "usage":{
                        "include":True
                    }
                }
            )
        # Add the assistant's response to messages
        messages.append(response.choices[0].message.model_dump())
        print("Input tokens:",response.usage.prompt_tokens)
        print("Output tokens:",response.usage.completion_tokens)
        return response

    except Exception as e:
        logging.error(f"Error occurred while calling LLM: {e}. Retrying request.")

        if hasattr(e, "response"):
            logging.info("Full HTTP response content:", e.response.text)

        return call_llm(messages=messages, model=model, provider=provider)


def execute_tool_call(tool_call):
    """
    Execute a tool call and return the result.
    
    Args:
        tool_call: The tool call object from the LLM response
    
    Returns:
        dict: Tool response message for the conversation
    """
    # print(tool_call)## need to comment this out
    try:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        logging.info(f"Executing tool: {tool_name} with args: {tool_args}")
        
        # Execute the tool function
        tool_result = TOOL_MAPPING[tool_name](**tool_args)
        
        tool_success = True
        full_tool_details = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": json.dumps(tool_result, indent=2)
        }
        return full_tool_details, tool_success
    except Exception as e:
        logging.error(f"Error occurred while executing tool: {e}")
        tool_success = False
        full_tool_details = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": "NONE - ERRONEOUS TOOL CALL"
        }
        return full_tool_details, tool_success


def run_agentic_loop(user_query, model="google/gemini-2.5-flash", provider=None, max_iterations=10):
    """
    Run an agentic loop that can use tools to create summaries of relevant information to user queries.
    
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
            Keep searching with increasing offset values until the returned actions become irrelevant to the user's query - this ensures you find all pertinent information before compiling your final summary.
            Once you have identified a list or batch of action documents that you want to expand fully, you should make multiple **parallel** calls to the get_action_details tool — each call should contain one of the action IDs from the batch.
            
            Then use the information you have gathered to create a summary of information which is relevant to the user's query.
            Do NOT use your own knowledge, make up statistics or state information that is not present in the action documents. ALL of your statements must be DIRECTLY supported by the evidence you found in the action documents. 
            This means for every statement you make you MUST cite the action id used to make that statement as a reference.
            You must NOT answer the question yourself - your job is just to find all the evidence in the action documents which are related to the user's query and summarise these into a concise paragraph.
            This means you MUST NOT use your own knowledge, add qualifying statements, draw conclusions or make your own judgements of the action information. You must only list relevant information from the action documents WITHOUT qualifying statements.
            
            Finally, you MUST use the get_formatted_result tool. This will convert the user's query, your generated summary and the list of the action IDs you used to generate your summary into a valid JSON format.

            You can only send ten messages to the system in total, and one message must be used to call the get_formatted_result tool. Parallel tool calls count as one message.
            Use these ten messages wisely, you should call get_formatted_result with your final summary before the ten messages have run out.
            """
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

        logging.info(f"Iteration {iteration_count}: Calling LLM...")
        print(f"Iteration {iteration_count}: Calling LLM...")
        
        try:
            response = call_llm(messages, model, provider)
        except FatalAPIError as e:
            logging.error(str(e))
            raise

        logging.info(f"Provider: {response.provider}")
        
        # Check if the model wants to use tools
        if response.choices[0].message.tool_calls:
            logging.debug("Processing tool calls...")
            logging.info(f"LLM requested {len(response.choices[0].message.tool_calls)} tool call(s)")
            
            # Execute all tool calls
            tool_messages = []
            for tool_call in response.choices[0].message.tool_calls:

                tool_response, tool_execution_success = execute_tool_call(tool_call)
                single_tool_call_info = {"function_name":tool_call.function.name, "args":tool_call.function.arguments, "return_val":tool_response["content"]}
                iteration_tool_calls["tools_called"].append(single_tool_call_info)

                if tool_execution_success:
                    logging.info(f"Tool execution successful. Tool name {tool_call.function.name}. Tool args: {tool_call.function.arguments}")

                    if tool_call.function.name == "get_formatted_result":
                        # The last execution of the tool call gave us the formatted final response 
                        result = json.loads(tool_response["content"])
                        logging.info("Conversation complete, with formatting tool call.")
                        logging.info(f"Final result: {result} ")
                        return result, all_tool_calls # SUCCESSFUL EXIT
                    else:
                        tool_messages.append(tool_response)
                else:
                    logging.error(f"Tool execution failed. Tool name {tool_call.function.name}. Tool args: {tool_call.function.arguments}")
                    messages.pop() # remove the assistant message that called the tool, so we can retry
                    tool_messages = [] # remove the results of even successfully executed tool calls from the last assistant message
                    break 

            messages.extend(tool_messages) # add results of tool calls to list of messages so far
            
        # else:
        #     No more tool calls, but this is not a valid exit - we need the llm to call get_formatted_result. Continue the loop until max iterations reached.
    
    if iteration_count >= max_iterations:
        logging.warning("Warning: Maximum iterations reached.")
    
        # Return the final assistant message
    final_message = messages[-1] if messages[-1]["role"] == "assistant" else messages[-2]
    final_message_content = final_message.get("content", "No response generated")
    logging.info(f"Final response (after hitting max iterations): {final_message_content}")
    print(f"Max iterations hit, final response: {final_message_content}")
    return final_message_content, all_tool_calls# FAILED EXIT


def get_prev_summaries(filename):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as file:
                summary_list = json.load(file)
                if not isinstance(summary_list, list):
                    raise RetrievalError(f"Expected JSON file to contain a list: {filename}")
                else:
                    logging.info(f"Loaded existing summaries from {filename}")
                    return summary_list
        except json.JSONDecodeError as e:
            raise RetrievalError(f"Failed to load JSON from existing summaries file: {filename}. Error: {e}")
    else:
        logging.info(f"Creating new summary output file {filename}.")
        return []
    

def assemble_summary_details(qu_details : dict, llm_response : dict, tool_use_track, model, provider="unpinned"):
    print("assembling summary")
    
    if isinstance(llm_response, dict):
        print("llm response is a dict")
        if qu_details["question"] != llm_response["query"]:
            logging.warning(f"The question in the LLM response does not match the original question, assembling erroneous summary details. Original question:{qu_details['question']}. Question in LLM response: {llm_response['query']}.")
            llm_response_details = {
                "relevant_summary": None,
                "summary_action_ids": None
            }
        else:
            llm_response_details = {
                "relevant_summary": llm_response["relevant_summary"],
                "summary_action_ids": llm_response["action_ids"],
            }
    
    else:
        print("llm response is not a dict")
        logging.warning("LLM response not formatted formatted properly, assembling erroneous summary details")
        print("LLM response not formatted formatted properly, assembling erroneous summary details")
        llm_response_details = {
            "relevant_summary": None,
            "summary_action_ids": None
        }
    
    summary_details = {
        "query": qu_details["question"],
        "model": model,
        "provider": provider, 
        "relevant_summary": llm_response_details["relevant_summary"],
        "summary_action_ids": llm_response_details["summary_action_ids"],           
        "tool_call_details": tool_use_track,
        "all_relevant_action_ids": qu_details["all_relevant_action_ids"],
        "regenerated_ids": qu_details.get("regenerated_ids", []),
    }
    return summary_details


def write_to_json_file(data_list, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data_list, file, indent=2)
    except json.JSONDecodeError as e:
        logging.error(f"Error writing to JSON file {filename}: {e}")


def append_new_summary(summary, filename, reset_file = False):
    if not reset_file:
        try:
            all_summaries = get_prev_summaries(filename=filename)
        except RetrievalError as e:
            logging.error(f"Could not load existing summaries, so failed writing new summary to file {filename}. Error: {e}")
            return
        all_summaries.append(summary)
    else:
        all_summaries = [summary]

    write_to_json_file(data_list=all_summaries, filename=filename)
    logging.info(f"Updated {filename} with new summary.")


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
    model_summaries = []
    for model, provider in model_provider_list:
        response, tool_calls = run_agentic_loop(user_query=query, model=model, provider=provider)
        model_summaries.append((model, provider, response, tool_calls))
    return model_summaries


def get_questions_from_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            qu_dicts = json.load(file)
            if not isinstance(qu_dicts, list):
                raise RetrievalError(f"Expected JSON file {filename} to contain a list, but contained {type(qu_dicts)} instead.")
            else:
                logging.info(f"Loaded questions from {filename}")
                return qu_dicts 
    except json.JSONDecodeError as e:
        raise RetrievalError(f"Error reading json from file {filename}: {str(e)}.")
    except FileNotFoundError:
        raise RetrievalError(f"Questions file {filename} not found.")
    

def run_summary_gen_for_qu_file(queries_filepath, max_qus, summary_out_base_dir, summary_filename, model_provider_list):
    try:
        file_qu_dicts = get_questions_from_file(queries_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load questions for summary generation from file {queries_filepath}: {e}")
        return

    qu_dicts = copy.deepcopy(file_qu_dicts)
    qu_count = 0
    current_qu_idx = -1

    while qu_count < max_qus:
        current_qu_idx += 1
        if current_qu_idx >= len(qu_dicts):
            logging.info(f"Reached end of questions in file {queries_filepath}. Stopping summary generation for this file.")
            break
        qu_dict = qu_dicts[current_qu_idx]
        query = qu_dict["question"]

        used_by_models = qu_dict.get("used_by_models", [])
        unused_by_models = [mp for mp in model_provider_list if list(mp) not in used_by_models]
        if unused_by_models:
            logging.info(f"Generating summaries for query: {query}")
            model_summaries = run_models(query=query, model_provider_list=unused_by_models)

            for model, provider, response, tool_calls in model_summaries:
                cleaned_model_name = parse_model_name(model)
                cleaned_provider_name = parse_provider_name(provider)
                summary_out_filepath = os.path.join(summary_out_base_dir, f"{cleaned_provider_name}_{cleaned_model_name}", summary_filename)
                assembled_summary = assemble_summary_details(qu_details=qu_dict, llm_response=response, tool_use_track=tool_calls, model=model, provider=provider)
                append_new_summary(summary=assembled_summary, filename=summary_out_filepath)

            qu_dict["used_by_models"] = used_by_models + unused_by_models
            qu_count += 1

    # overwrite the question file (it will contain the updated used_by_models field)
    write_to_json_file(data_list=qu_dicts, filename=queries_filepath)


def run_summary_gen_for_qu_dir(qus_dir, model_provider_list, summary_out_base_dir, offset_to_first_qu_file=0, max_qu_files=1, max_qus=1): 
    if not os.path.exists(qus_dir):
        logging.error(f"Questions directory {qus_dir} does not exist.")
        return
    else:
        logging.info(f"Starting summary generation for questions in directory {qus_dir}.")
        qus_filenames = []
        for qus_filename in sorted(os.listdir(qus_dir)):
            if qus_filename.endswith(".json"):
                qus_filenames.append(qus_filename)

        for qus_filename in qus_filenames[offset_to_first_qu_file : offset_to_first_qu_file + max_qu_files]:
            if qus_filename.endswith(".json"):
                retrieval_subdir = f"{RETRIEVAL_TYPE}" if RETRIEVAL_TYPE != "hybrid" else f"{RETRIEVAL_TYPE}_{FUSION_TYPE.replace(' ','-')}"
                filename_list = os.path.splitext(qus_filename)[0].split("_")
                filename_list[-1] = "summaries"
                summary_filename = "_".join(filename_list) + ".json"
                run_summary_gen_for_qu_file(
                    queries_filepath=os.path.join(qus_dir, qus_filename),
                    max_qus=max_qus,
                    summary_out_base_dir=os.path.join(summary_out_base_dir, retrieval_subdir), 
                    summary_filename=summary_filename, 
                    model_provider_list=model_provider_list
                )
            


def main():
    logging.basicConfig(filename = "logfiles/summary_gen_parallel.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # disable httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # disable bm25s logging
    logging.getLogger("bm25s").setLevel(logging.WARNING)

    model_provider_list = [
        ("openai/gpt-5", None),
        ("anthropic/claude-sonnet-4", None),
        ("google/gemini-2.5-pro", None),
        ("moonshotai/kimi-k2-0905", "fireworks/fp8")
    ]

    ## SUMMARY GENERATION PROCESS
    logging.info("STARTING summary generation process.")
    start_time = time.monotonic()
    qu_type = "answerable" # options: "answerable", "unanswerable"
    filter_stage = "passed" # options: "passed", "failed"
    qus_dir = f"live_questions/bg_km_qus/{qu_type}/{filter_stage}/usage_annotated"
    summary_out_base_dir = f"summary_gen_data/{qu_type}_{filter_stage}_qus_summaries"
    offset = 8
    max_qu_files = 25
    max_qus = 3

    try:
        run_summary_gen_for_qu_dir(
            qus_dir = qus_dir,
            model_provider_list=model_provider_list, 
            summary_out_base_dir=summary_out_base_dir, 
            max_qu_files=max_qu_files,
            offset_to_first_qu_file=offset,
            max_qus=max_qus
        )
    except KeyboardInterrupt as e:
        logging.error(f"Keyboard interrupt: {e}")
    end_time = time.monotonic() - start_time
    print("Time taken:",end_time)
    logging.info("ENDED summary generation process.")



if __name__ == "__main__":
    main()