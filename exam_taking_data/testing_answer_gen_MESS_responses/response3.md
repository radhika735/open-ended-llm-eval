# Comments: #
Model: gemini-2.5-flash
Telling it in the sys prompt that it actually needs to answer the user's question worked, it is actually generating an answer.
But only drawing upon one relevant action - need to get it to use more.

Trying adding this line: 

# Prompt: #
user_query = "What are the most effective interventions for reducing bat fatalities at wind turbines?"

messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that can answer conservation based questions by searching through conservation action documents to find relevant information required to answer the question you are asked.
            Use the search_actions tool to find relevant actions based on user queries, and then use get_action_details to retrieve full details for specific actions when needed. 
            The full details of a particular action will contain the action name, the action ID, outcomes of studies researching that action, and the effectiveness of the action based on the information from these supporting studies.
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
    


# Response: #
Final response: {
  "action_ids": [
    "24"
  ],
  "final_answer_to_query": "The most effective intervention for reducing bat fatalities at wind turbines is to increase the cut-in speed of the turbines. This means the turbines start rotating at a higher wind speed than typical. Research indicates that increasing the cut-in speed by 1.0 m/s can reduce bat fatalities by 49-92%. This also results in a minimal loss of Annual Energy Production, typically between 0.3% and 1.0%. (Action 24)"
}