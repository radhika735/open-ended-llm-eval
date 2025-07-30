# Comments: #
Model: gemini-2.5-flash
Told it in the sys prompt that the full details of the action documents include their name, id, outcomes of studies researching that action, and the effectiveness of the action. Worked, because it does think it can answer the question now as compared to before where it thought it couldn't.
Need to get it to actually answer the question though!


# Prompt: #
user_query = "What are the most effective interventions for reducing bat fatalities at wind turbines?"

messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that can search through conservation action documents to find relevant information.
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
  "action_ids": [],
  "final_answer_to_query": "I can help you with this. I will look for actions related to reducing bat fatalities at wind turbines. Please wait while I retrieve the information."
}