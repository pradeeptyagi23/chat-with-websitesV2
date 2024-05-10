import requests

inputs = {"input":{"input": "what do you know about harrison", "chat_history": []}}
response = requests.post("http://localhost:8000/rag-chroma/invoke", json=inputs)

print(response.json())