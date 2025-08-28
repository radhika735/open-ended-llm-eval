# import required modules
from ragas.metrics import Faithfulness, AnswerRelevancy, FaithfulnesswithHHEM
from ragas import SingleTurnSample
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os 
import json
from dotenv import load_dotenv
import asyncio

# loading environment variables
load_dotenv()


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data     


def get_retrieved_contexts(answer_dict):
    contexts = []
    tool_calls = answer_dict.get("tool_calls", [])

    for iteration in tool_calls:
        iteration_tools = iteration.get("tools_called", [])
        for tool in iteration_tools:
            if tool.get("function_name") in ["search_actions", "get_action_details"]:
                context = tool.get("return_val", "")
                if context != "":
                    contexts.append(context)

    return contexts



# get data
def get_data(filename):
    file_answers = read_json_file(filename)
    answer_to_evaluate = file_answers[0]
    user_input = answer_to_evaluate["query"]
    response = answer_to_evaluate["answer"]
    retrieved_contexts = get_retrieved_contexts(answer_to_evaluate)
    data = {
        "user_input": user_input,
        "response": response,
        "retrieved_contexts": retrieved_contexts
    }
    return data


# evaluate
def get_ragas_evaluation(data):
    result = evaluate (
        data,
        metrics =[
            faithfulness ,
            answer_relevancy ,
        ],
    )
    print(result)
    return result


async def test(filename):
    test_data = get_data(filename)
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            # model = "deepseek/deepseek-r1-0528:free"
            model="qwen/qwen3-235b-a22b-2507"
        )
    )
    sample = SingleTurnSample(**test_data)
    scorer = Faithfulness(llm=evaluator_llm)
    hhem_scorer = FaithfulnesswithHHEM(llm=evaluator_llm)
    #result = await scorer.single_turn_ascore(sample)
    #result = await hhem_scorer.single_turn_ascore(sample)
    statements = await score._
    return result



def main():
    get_ragas_evaluation()


if __name__ == "__main__":
    #data = get_data("answer_gen_data/experimental/parallel_tools/answers__claude-opus-4.json")
    eval = asyncio.run(test("answer_gen_data/experimental/parallel_tools/answers__claude-opus-4.json"))
    print(eval)