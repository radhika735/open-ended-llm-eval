# Comments: #
Passing it only the km action files, both with and without evidence for a single synopsis, and asking it to generate questions related to that synopsis.
Not few-shot prompting it yet with example questions on other species, so it doesn't think of asking about other species (as questions about another species may be answerable if we had given it the entire database). Need to test effect of giving it questions from the same species.
Do we even want to limit the questions generated to be about a particular species? Will that throw off answer generation quite a bit? Also it won't allow for even broader scale questions.
Not sure if including the potential question types has any benefit.


# Prompt: #
Attached is a file containing conservation actions, their effectiveness and key messages relating to them. This has been gathered by hand and forms part of the Conservation Evidence living evidence database. Each action is prefixed with a numerical id.
You will be generating synthetic questions for use in an experiment. Generate four questions which CANNOT be answered by any of the actions in this database. Keep the questions short. Include a model answer to this question, explaining clearly why the question is unanswerable.

Here are some example reasons why a particular question may be unanswerable:
a) Answering the question requires drawing on actions for which there has been no supporting evidence.
b) The question asks about a factor which is not recorded in the database (e.g. cost-effectiveness)
c) None of the actions in the database are relevant enough to answer the question.
d) The question is too broad / not relevant to the database.

Potential question types:
a) Effects/outcomes of specific conservation actions
b) Actions to achieve specific conservation objectives/outcomes
c) Alternatives to achieve a particular conservation goal
d) Trade-offs between different conservation actions
e) Geographical and contextual variations in action effectiveness
f) Scaling of actions and their impacts
g) Timeframes for expected outcomes
h) Factors associated with success or failure of conservation efforts

Generate questions about Amphibian Conservation.


# Response: #
Here are four questions that cannot be answered by the provided text, along with model answers explaining why.

Question 1:
What is the most cost-effective method for creating new pond habitats for great crested newts in the UK?

Model Answer:
This question cannot be answered because the database does not contain information on the financial costs of conservation actions. While several actions describe the creation of ponds for great crested newts (e.g., action 863) and are rated as 'Likely to be beneficial', the key messages only summarise biological outcomes like population establishment and colonisation rates. There is no data provided on the implementation costs (e.g., excavation, planting, labour) or long-term maintenance expenses for these actions, which would be essential for a cost-effectiveness analysis.

Question 2:
What are the impacts of noise pollution from major highways on amphibian breeding success?

Model Answer:
This question cannot be answered as none of the actions in the database investigate the effects of noise pollution on amphibians. The evidence base addresses threats from roads in terms of direct mortality and habitat fragmentation, proposing solutions like culverts (884), barrier fences (756), and road closures (842). However, there are no studies mentioned that measure the specific impact of traffic-related noise pollution on amphibian behaviour, communication, or reproductive outcomes.

Question 3:
Which conservation strategies are most effective for threatened caecilian species in South America?

Model Answer:
This question is unanswerable because the provided evidence base contains no information on conservation actions for caecilians. The actions and studies listed focus almost exclusively on anurans (frogs and toads) and caudates (salamanders and newts), with the vast majority of studies conducted in North America and Europe. There are no actions relevant to the specific biological needs or threats facing caecilians, nor are there any studies from South America that address this specific order of amphibians.

Question 4:
How will projected climate change-induced temperature increases affect the long-term success of captive-breeding and release programmes in Australia?

Model Answer:
This question cannot be answered because it requires a forward-looking analysis that synthesizes climate change models with conservation outcomes, which is beyond the scope of the provided historical evidence. The database contains information on the outcomes of past and current captive-breeding programmes in Australia (e.g., actions 870, 872) and the effect of temperature on captive animals (e.g., 835, 770). However, it does not contain any studies that model or predict how future climate scenarios might impact the survival, adaptation, or establishment of released populations in the wild.