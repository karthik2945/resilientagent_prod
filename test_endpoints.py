from fastapi.testclient import TestClient
from server.app import app
import json

client = TestClient(app)

# Test 1: Empty body (like openenv validator sends when it sends no json)
resp = client.post("/reset")
print("Response without body:", resp.status_code)
if resp.status_code != 200:
    print(resp.json())

# Test 2: 'null' JSON body
resp = client.post("/reset", json=None)
print("Response with null json:", resp.status_code)
if resp.status_code != 200:
    print(resp.json())

# Test 3: empty dict JSON body
resp = client.post("/reset", json={})
print("Response with empty dict json:", resp.status_code)

# Test 4: actual task_id
resp = client.post("/reset", json={"task_id": "task3_cascading_failure"})
print("Response with actual json:", resp.status_code)
print("Returned task_id observation metrics...", "task3" in str(resp.json()))
