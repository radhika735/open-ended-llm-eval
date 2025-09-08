import bm25s
import os
import logging
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
# from pydantic import BaseModel, Field
# from typing import Annotated
from utils.action_retrieval import ActionRetrievalContext, RetrievalError, get_parsed_action_by_id, sparse_retrieve_docs, dense_retrieve_docs, hybrid_retrieve_docs


load_dotenv()


ACTION_RETRIEVAL_CONTEXT = ActionRetrievalContext(required_fields=["action_id", "action_title", "key_messages"])
RETRIEVAL_TYPE = "sparse" # other options: "dense", "hybrid".
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
        return sparse_retrieve_docs(query_string=query_string, context=ACTION_RETRIEVAL_CONTEXT, k=k, offset=offset)
    elif RETRIEVAL_TYPE == "dense":
        return dense_retrieve_docs(query_string=query_string, context=ACTION_RETRIEVAL_CONTEXT, k=k, offset=offset)
    elif RETRIEVAL_TYPE == "hybrid":
        if FUSION_TYPE not in ["cross-encoder", "reciprocal rank fusion"]:
            logging.warning("Invalid FUSION_TYPE set for hybrid retrieval of action documents. Must be either 'cross-encoder' or 'reciprocal rank fusion'. Defaulting to using cross-encoder.")
            fusion_type = "cross-encoder"
        else:
            fusion_type = FUSION_TYPE
        return hybrid_retrieve_docs(query_string=query_string, context=ACTION_RETRIEVAL_CONTEXT, fusion_type=fusion_type, k=k, offset=offset)
    else:
        logging.warning("Invalid RETRIEVAL_TYPE set for retrieving action documents by similarity to query string. Must be either 'sparse', 'dense' or 'hybrid'. Defaulting to sparse retrieval.")
        return sparse_retrieve_docs(query_string=query_string, context=ACTION_RETRIEVAL_CONTEXT, k=k, offset=offset)


def get_action_details(action_id):
    """
    Retrieve the full details for a specific action by its ID.
    
    Args:
        action_id (str): The action ID to retrieve (e.g., "1", "101", "1002")
    
    Returns:
        dict: Full action details or None if not found
    """
    # parsed_actions = get_all_parsed_actions(context=ACTION_RETRIEVAL_CONTEXT)
    
    # # Find the action with matching ID
    # for action in parsed_actions:
    #     if action["action_id"] == action_id:
    #         logging.debug(f"Found action details for ID: {action_id}")
    #         return action
    # logging.debug(f"Action ID {action_id} not found")
    # return {
    #     "error": f"Action with ID '{action_id}' not found",
    #     "available_ids": [action["action_id"] for action in parsed_actions[:10]]  # Show first 10 as examples
    # }
    parsed_action = get_parsed_action_by_id(id=action_id, context=ACTION_RETRIEVAL_CONTEXT)
    if parsed_action is not None:
        logging.debug(f"Found action details for ID: {action_id}")
        return parsed_action
    else:
        logging.debug(f"Action ID {action_id} not found")
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
                    "require_parameters": True
                }
            )
        # Add the assistant's response to messages
        messages.append(response.choices[0].message.model_dump())
        print("\n\n\nResponse:",response,"\n\n\n")
        print("Messages:", messages, "\n\n\n")
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
    print(tool_call)## need to comment this out
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

        logging.debug(f"Iteration {iteration_count}: Calling LLM...")
        
        response = call_llm(messages, model, provider)
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
                    logging.debug(f"Tool execution successful. Tool name {tool_call.function.name}. Tool args: {tool_call.function.arguments}")

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
        #     # No more tool calls, we have the final response
        #     logging.warning("Conversation complete, without formatting tool call.")
        #     logging.info(f"Final response: {response.choices[0].message.content}")
        #     break
    
    if iteration_count >= max_iterations:
        logging.warning("Warning: Maximum iterations reached")
    
        # Return the final assistant message
    final_message = messages[-1] if messages[-1]["role"] == "assistant" else messages[-2]
    final_message_content = final_message.get("content", "No response generated")
    logging.info(f"Final response (after hitting max iterations): {final_message_content}")
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




def assemble_llm_response_and_tools(response : dict, tool_use_track):
    assembled_response = {}
    # response is a dict with the properties 
    if isinstance(response, dict):
        assembled_response.update(response)
        assembled_response["tool_call_details"] = tool_use_track
        return assembled_response
    else:
        logging.warning("LLM response not formatted, unable to assemble response")
        raise TypeError(f"Expected LLM response to be formatted as a dictionary, instead it is a {type(response)}. Unable to assemble final response")
    



def write_new_summaries(summary_list, filename, reset_file = False):
    if not reset_file:
        try:
            all_summaries = get_prev_summaries(filename=filename)
        except RetrievalError as e:
            logging.error(f"Could not load existing summaries, so failed writing new summaries to file {filename}. Error: {e}")
            return
        all_summaries.extend(summary_list)
    else:
        all_summaries = summary_list

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(all_summaries, file, indent=2)
    logging.info(f"Updated {filename} with new summaries.")



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



def run_models(query, model_provider_list, summary_out_dir):
    for model, provider in model_provider_list:
        cleaned_model_name = parse_model_name(model)
        cleaned_provider_name = parse_provider_name(provider)
        filename = f"summaries_{RETRIEVAL_TYPE}_{cleaned_provider_name}_{cleaned_model_name}.json"
        summary_out_file = os.path.join(summary_out_dir, filename)
        response, tool_calls = run_agentic_loop(user_query=query, model=model, provider=provider)
        write_new_summaries(summary_list=[assemble_llm_response_and_tools(response, tool_calls)], filename=summary_out_file)



def run_queries_on_models(query_list, model_provider_list, summary_out_dir):
    for query in query_list:
        run_models(query=query, model_provider_list=model_provider_list, summary_out_dir=summary_out_dir)



def get_questions_from_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            qu_dicts = json.load(file)
            success = True
            logging.info(f"Loaded questions from {filename}")  
    except json.JSONDecodeError as e:
        qu_dicts = []
        success = False
        logging.warning(f"Failed to load questions from {filename}.")
    except FileNotFoundError:
        qu_dicts = []
        success = False
        logging.warning(f"Questions file {filename} not found.")

    questions = [qu["question"] for qu in qu_dicts]

    return success, questions



def run_summary_gen_for_qu_dir(unused_qus_dir, used_qus_dir, model_provider_list, summary_out_base_dir):
    # questions = get_questions_from_directory(directory=qu_dir)
    for filename in os.listdir(unused_qus_dir):
        if filename.endswith(".json"):
            success, questions = get_questions_from_file(os.path.join(unused_qus_dir, filename))
            if success:
                summary_out_sub_dir = os.path.splitext(filename)[0]
                summary_out_dir = os.path.join(summary_out_base_dir, summary_out_sub_dir)
                run_queries_on_models(query_list=questions, model_provider_list=model_provider_list, summary_out_dir=summary_out_dir)


def main():
    logging.basicConfig(filename = "logfiles/summary_gen.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # disable httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # disable bm25s logging
    logging.getLogger("bm25s").setLevel(logging.WARNING)

    ## USEFUL/USED QUERIES TO KEEP TRACK OF
    queries = [
        # "How effective is providing artificial nesting sites for different types of bees?",
        # "What are the documented trade-offs of using prescribed fire as a forest management tool?",
        # "How can altering the type of food provided to captive primates improve their welfare?",
        # "What are the most effective ways to modify man-made structures like windows and power lines to reduce bird collision mortality?",
        # "When creating new habitats on farmland, how long does it typically take to see benefits for wildlife?",
    
        # "What conservation actions are most beneficial for establishing new populations of threatened toad species in the UK?",
        
        # "What are the most effective interventions for reducing bat fatalities at wind turbines?",

        # "What are the most beneficial actions for reducing human-wildlife conflict with bears?",
        # "What actions can be taken to mitigate the environmental pollution caused by waste from salmon farms?", 
        # "What are the most effective ways to increase soil organic carbon on loamy soils?"
    ]

    # model_provider_list = [
    #     ("google/gemini-2.5-flash", None), # CHEAPISH
    #     ("moonshotai/kimi-k2", "fireworks/fp8"), # CHEAPISH
    #     ("qwen/qwen3-235b-a22b-thinking-2507", "together"), # CHEAP
        
    #     ("anthropic/claude-sonnet-4", None), # HIGH PRICE

    #     #("anthropic/claude-opus-4", None), # V. EXPENSIVE, AVOID!!
    #     #("x-ai/grok-4", None), # HIGH PRICE (AVOID?)
    #     ("openai/gpt-4.1", None), # MID PRICE
    #     ("google/gemini-2.5-pro", None), # MID PRICE
    #     ("deepseek/deepseek-r1-0528", "novita/fp8"), # CHEAP
    #     #("mistralai/magistral-medium-2506", None), # DOESN'T RLY WORK - UNPROCESSABLE ENTITY ERROR
    #     #("mistralai/magistral-medium-2506:thinking", None), # DOESN'T RLY WORK - UNPROCESSABLE ENTITY ERROR
    #     ("anthropic/claude-3.5-sonnet", None), # HIGH PRICE
    #     #("qwen/qwen-2.5-72b-instruct", "novita"), # DOESN'T RLY WORK - ARGS FOR TOOLS CALLS MALFORMED

    #     ("openai/gpt-5", None) # MID PRICE

    # ]
    # could also test gemini-2.5-flash-lite


    model_provider_list = [
        ("openai/gpt-5", None),
        ("anthropic/claude-sonnet-4", None),
        ("google/gemini-2.5-pro", None),
        ("moonshotai/kimi-k2", "fireworks/fp8")
    ]

    ## CREATING SUMMARIES FOR MINI LLM JUDGE EVAL QUESTIONS
    logging.info("STARTING summary generation process.")
    #qu_retrieval_success, test_questions = get_questions_from_file("evaluation_data/mini_testing_human_agreement/test_questions.json")
    # test_questions = [
    #     "What are the most beneficial actions for reducing human-wildlife conflict with bears?",
    #     "What actions can be taken to mitigate the environmental pollution caused by waste from salmon farms?", 
    #     "What are the most effective ways to increase soil organic carbon on loamy soils?",
    #     "What are the most effective interventions for reducing bat fatalities at wind turbines?"
    # ]
    # run_queries_on_models(query_list=test_questions, model_provider_list=model_provider_list, summary_out_dir="answer_gen_data/without_effectiveness_summaries/v4_prompt")
    unused_qus_dir = "summary_gen_data/bg_km_qus_unused/answerable/passed"
    used_qus_dir = "summary_gen_data/bg_km_qus_used/answerable/passed"
    summary_out_base_dir = "summary_gen_data/passed_answerable_qus_summaries"
    run_summary_gen_for_qu_dir(unused_qus_dir=unused_qus_dir, used_qus_dir=used_qus_dir, model_provider_list=model_provider_list, summary_out_base_dir=summary_out_base_dir)
    logging.info("ENDED summary generation process.")





if __name__ == "__main__":
    main()