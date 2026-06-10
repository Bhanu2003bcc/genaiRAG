# pyrefly: ignore [missing-import]
from locust import HttpUser, task, between

class RAGUser(HttpUser):
    # Simulated users wait 1 to 3 seconds between actions
    wait_time = between(1, 3)
    
    # Store token after login
    token = None

    def on_start(self):
        """ Runs when a simulated user is created. Handles registration and login/auth. """
        import uuid
        import random
        import time
        # Sleep randomly up to 4 seconds to stagger registrations and logins, avoiding CPU bcrypt spikes
        time.sleep(random.uniform(0, 4))
        
        uid = uuid.uuid4().hex[:8]
        username = f"user_{uid}"
        email = f"{uid}@example.com"
        password = "TestPass123"

        reg_payload = {
            "username": username,
            "email": email,
            "password": password
        }
        self.client.post("/api/auth/register", json=reg_payload)
        
        login_payload = {
            "username": username,
            "password": password
        }
        response = self.client.post("/api/auth/login", json=login_payload)
        if response.status_code == 200:
            self.token = response.json().get("data", {}).get("token")

    @task(3)
    def view_documents(self):
        """ Read-heavy cached endpoint """
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.client.get("/api/documents?page=0&size=20", headers=headers)

    @task(2)
    def view_conversations(self):
        """ Read-heavy cached endpoint """
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.client.get("/api/chat/conversations?page=0&size=20", headers=headers)

    @task(1)
    def ask_question(self):
        """ Write-heavy DB and LLM query """
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        payload = {"message": "Tell me about the Eiffel Tower."}
        self.client.post("/api/chat", json=payload, headers=headers)
