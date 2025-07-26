import requests
import json

# Test data
url = "http://localhost:8080/hackrx/run"
headers = {
    "Authorization": "Bearer testtoken",
    "Content-Type": "application/json"
}
data = {
    "documents": "https://hackrx.in/policies/BAJHLIP23020V012223.pdf",
    "questions": [
        "What is the waiting period for pre-existing diseases?",
        "What does this policy cover?"
    ]
}

# Send request
try:
    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
