# Comments: #
Model = kimi k2
provider = baseten
Not calling the tools + not answering the question. Maybe not  explaining how to use default parameters of search_actions() is throwing it off?

# Prompt: #
user_query = "What are the most effective interventions for reducing bat fatalities at wind turbines?"
messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant tasked with answering queries about conservation actions.
            You can find relevant information required to answer the question by searching through conservation action documents.
            You MUST use both available tools - 'search_actions' and 'get_action_details' - at least once for every query 
            even if you already know the answer.
            It is of vital importance that all information you state comes from action documents. 
            You MUST NOT answer the question using your own knowledge.

            First, use 'search_actions' to find relevant actions based on the query. 
            Then, use 'get_action_details' to retrieve details for each action you found. 
            'get_action_details' will give you information about outcomes of studies investigating that action, and the action's effectiveness.

            Your final answer to the query must be based on the information retrieved from both tools.
            Cite the ids of the relevant actions you use in your final response.
            
            Return your answer in structured JSON, including the answer and a list of the IDS of the actions used. 
            """
        },
        {
            "role": "user",
            "content": user_query,
        }
    ]
    



# Tools: #
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_actions",
            "description": "Search for conservation actions relevant to the user's query. Always use this tool first to find possible actions.",
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
            "description": "Retrieve detailed information about a specific conservation action using its ID. Use this tool for each action found by 'search_actions'.",
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

Tool mapping for function execution
TOOL_MAPPING = {
    "search_actions": search_actions,
    "get_action_details": get_action_details
}




# Response: #
Final response: {"action_ids":[],"final_answer_to_query":"I will search for conservation actions specifically targeting bat fatalities at wind turbines to provide you with evidence-based interventions. I am using the query: 'reducing bat fatalities at wind turbines' to find relevant actions and then retrieve their effectiveness details using the provided tools."}



# Other client setup: #
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

client = get_client()
    response = client.chat.completions.create(
        model=model,
        tools=tools,
        response_format=response_format,
        messages=messages,
        extra_body={
            "require_parameters": True,
            "provider": {
                "order": ["deepinfra/fp4"], # Specify the single provider you want to pin
                #"allow_fallbacks": False     # Set fallback to None to prevent routing elsewhere
            }
            
        }
    )