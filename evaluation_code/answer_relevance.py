import logging
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_client():
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    return client

def call_llm():
    messages = [
        {
            ""
        }
    ]







if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="logfiles/answer_relevance.log")