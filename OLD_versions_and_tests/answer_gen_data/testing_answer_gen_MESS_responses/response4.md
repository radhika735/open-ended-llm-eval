# Comments: #
Model: gemini-2.5-flash
Did get response including multiple actions. 
But response did not prioritise good answering techniques i.e. should mention the MOST effective interventions first.
Also just noticed that the action ids it is returning are nonsensical...
  Just noticed that get_parsed_actions() was incorrectly parsing actions and returning zero actions, i.e. the llm would have had access to no actions and so was probably using parametric memory instead of the actual database. This would explain the nonsensical action ids.
  Have fixed get_parsed_actions() to be correct now so hopefully this is fixed. But may also be good to include a line in the prompt that says something like "do not state facts / information that you have not found from the database".

# Prompt: #
user_query = "What are the most effective interventions for reducing bat fatalities at wind turbines?"

messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that can answer conservation based questions by searching through conservation action documents to find relevant information required to answer the question you are asked.
            Use the search_actions tool to find relevant actions based on user queries, and then use get_action_details to retrieve full details for specific actions when needed. 
            The full details of a particular action will contain the action name, the action ID, outcomes of studies investigating that action, and the effectiveness of the action based on the information from these supporting studies.
            You need to answer the user's question using ALL action documents that are sufficiently relevant, not just the most relevant one.
            When searching for actions, start with the default parameters, and if you need more results, use the offset parameter to retrieve additional actions. 
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
2025-07-29 11:33:39,328 - INFO - Final response: {
  "action_ids": [
    "48",
    "60",
    "61",
    "63"
  ],
  "final_answer_to_query": "The most effective interventions for reducing bat fatalities at wind turbines are ultrasonic acoustic deterrents, reducing cut-in speed, and feathering turbine blades at low wind speeds. Ultrasonic acoustic deterrents (Action 48) have shown moderate effectiveness, reducing bat activity and fatalities with no obvious sign of habituation. Reducing cut-in speed (Action 60) and feathering turbine blades at low wind speeds (Action 61) are highly effective, reducing bat fatalities by over 50%. These actions are particularly important during periods of high bat activity and low wind speeds, typically from July through October. Other actions, such as acoustic alarms, are still being tested and their effectiveness is yet to be determined (Action 63)."
}
