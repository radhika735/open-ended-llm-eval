import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Make sure your environment variable is set
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in your environment.")

# Initialize client
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "Authorization": f"Bearer {api_key}"
    }
)

# Minimal call to test key validity
# try:
#     models = client.models.list()
#     print("✅ API key is valid. Available models:")
#     for model in models.data:
#         print("-", model.id)
# except Exception as e:
#     print("❌ API key is invalid or not authorized. Error:")
#     print(e)

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, test"}],
        max_tokens=50,
        timeout=30
    )
    print(response.choices[0].message.content)
except Exception as e:
    print("Error during chat completion:", e)