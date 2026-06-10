"""
Live API Test Suite for RAG System Backend
Tests all endpoints against the running server at http://localhost:8080
"""
import sys
import json
import time
import io
import httpx

BASE = "http://localhost:8080/api"
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
INFO = "\033[96mℹ INFO\033[0m"

results = []

def log(tag, name, msg="", body=None):
    status = "PASS" if tag == PASS else "FAIL" if tag == FAIL else "WARN"
    results.append({"status": status, "test": name, "msg": msg})
    print(f"  {tag}  {name}")
    if msg:
        print(f"         {msg}")
    if body and status == "FAIL":
        try:
            parsed = json.dumps(body, indent=8) if isinstance(body, dict) else str(body)[:300]
            print(f"         Response: {parsed}")
        except Exception:
            pass

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def run_tests():
    token = None
    doc_id = None
    conv_id = None
    ts = int(time.time())
    test_user = f"testuser_{ts}"
    test_email = f"test_{ts}@example.com"
    test_pass = "TestPass123"

    with httpx.Client(timeout=30) as client:

        # ──────────────────────────────────────────────────────
        section("1. HEALTH CHECK")
        # ──────────────────────────────────────────────────────
        try:
            r = client.get(f"{BASE}/actuator/health")
            if r.status_code == 200 and r.json().get("status") == "UP":
                log(PASS, "GET /actuator/health", f"status=UP")
            else:
                log(FAIL, "GET /actuator/health", f"HTTP {r.status_code} | {r.text[:100]}")
        except Exception as e:
            log(FAIL, "GET /actuator/health", str(e))

        # ──────────────────────────────────────────────────────
        section("2. AUTH — REGISTER")
        # ──────────────────────────────────────────────────────
        try:
            payload = {"username": test_user, "email": test_email, "password": test_pass}
            r = client.post(f"{BASE}/auth/register", json=payload)
            body = r.json()
            if r.status_code == 200 and body.get("success"):
                token = body["data"]["token"]
                log(PASS, "POST /auth/register", f"user={test_user} | token obtained")
            else:
                log(FAIL, "POST /auth/register", f"HTTP {r.status_code}", body)
        except Exception as e:
            log(FAIL, "POST /auth/register", str(e))

        # ──────────────────────────────────────────────────────
        section("3. AUTH — REGISTER VALIDATION ERRORS")
        # ──────────────────────────────────────────────────────
        try:
            r = client.post(f"{BASE}/auth/register", json={"username": "ab", "email": "bad", "password": "123"})
            if r.status_code == 400:
                log(PASS, "POST /auth/register (bad payload → 400)", f"Validation correctly rejected")
            else:
                log(WARN, "POST /auth/register (bad payload)", f"Expected 400, got {r.status_code}")
        except Exception as e:
            log(FAIL, "POST /auth/register (bad payload)", str(e))

        try:
            # Duplicate user
            payload = {"username": test_user, "email": test_email, "password": test_pass}
            r = client.post(f"{BASE}/auth/register", json=payload)
            if r.status_code in (400, 409):
                log(PASS, "POST /auth/register (duplicate → 4xx)", f"HTTP {r.status_code}")
            else:
                log(WARN, "POST /auth/register (duplicate)", f"Expected 4xx, got {r.status_code} | {r.text[:100]}")
        except Exception as e:
            log(FAIL, "POST /auth/register (duplicate)", str(e))

        # ──────────────────────────────────────────────────────
        section("4. AUTH — LOGIN")
        # ──────────────────────────────────────────────────────
        try:
            r = client.post(f"{BASE}/auth/login", json={"username": test_user, "password": test_pass})
            body = r.json()
            if r.status_code == 200 and body.get("success"):
                token = body["data"]["token"]  # refresh token
                log(PASS, "POST /auth/login", f"token refreshed")
            else:
                log(FAIL, "POST /auth/login", f"HTTP {r.status_code}", body)
        except Exception as e:
            log(FAIL, "POST /auth/login", str(e))

        try:
            r = client.post(f"{BASE}/auth/login", json={"username": test_user, "password": "wrongpassword"})
            if r.status_code in (400, 401, 403):
                log(PASS, "POST /auth/login (wrong password → 4xx)", f"HTTP {r.status_code}")
            else:
                log(WARN, "POST /auth/login (wrong password)", f"Expected 4xx, got {r.status_code}")
        except Exception as e:
            log(FAIL, "POST /auth/login (wrong password)", str(e))

        if not token:
            print(f"\n  {WARN}  No token obtained — skipping authenticated endpoints")
            return

        headers = {"Authorization": f"Bearer {token}"}

        # ──────────────────────────────────────────────────────
        section("5. AUTH — INVALID TOKEN")
        # ──────────────────────────────────────────────────────
        try:
            r = client.get(f"{BASE}/documents", headers={"Authorization": "Bearer invalidtoken123"})
            if r.status_code == 401:
                log(PASS, "GET /documents (bad token → 401)", f"HTTP 401")
            else:
                log(WARN, "GET /documents (bad token)", f"Expected 401, got {r.status_code}")
        except Exception as e:
            log(FAIL, "GET /documents (bad token)", str(e))

        # ──────────────────────────────────────────────────────
        section("6. DOCUMENTS — LIST (empty)")
        # ──────────────────────────────────────────────────────
        try:
            r = client.get(f"{BASE}/documents", headers=headers)
            body = r.json()
            if r.status_code == 200 and body.get("success"):
                count = body["data"]["totalElements"]
                log(PASS, "GET /documents", f"totalElements={count}")
            else:
                log(FAIL, "GET /documents", f"HTTP {r.status_code}", body)
        except Exception as e:
            log(FAIL, "GET /documents", str(e))

        # ──────────────────────────────────────────────────────
        section("7. DOCUMENTS — UPLOAD")
        # ──────────────────────────────────────────────────────
        txt_content = b"The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel. The tower stands 330 metres tall."
        try:
            r = client.post(
                f"{BASE}/documents/upload",
                files={"file": ("test_rag.txt", io.BytesIO(txt_content), "text/plain")},
                headers=headers,
            )
            body = r.json()
            if r.status_code == 200 and body.get("success"):
                doc_id = body["data"]["id"]
                log(PASS, "POST /documents/upload", f"doc_id={doc_id} | status={body['data']['status']}")
            else:
                log(FAIL, "POST /documents/upload", f"HTTP {r.status_code}", body)
        except Exception as e:
            log(FAIL, "POST /documents/upload", str(e))

        # ──────────────────────────────────────────────────────
        section("8. DOCUMENTS — GET BY ID")
        # ──────────────────────────────────────────────────────
        if doc_id:
            try:
                r = client.get(f"{BASE}/documents/{doc_id}", headers=headers)
                body = r.json()
                if r.status_code == 200 and body.get("success"):
                    log(PASS, f"GET /documents/{doc_id}", f"title={body['data']['title']} status={body['data']['status']}")
                else:
                    log(FAIL, f"GET /documents/{{doc_id}}", f"HTTP {r.status_code}", body)
            except Exception as e:
                log(FAIL, f"GET /documents/{{doc_id}}", str(e))
        else:
            log(WARN, "GET /documents/{doc_id}", "skipped — no doc_id from upload")

        # ──────────────────────────────────────────────────────
        section("9. DOCUMENTS — UPLOAD (unsupported type)")
        # ──────────────────────────────────────────────────────
        try:
            r = client.post(
                f"{BASE}/documents/upload",
                files={"file": ("malware.exe", io.BytesIO(b"\x4d\x5a\x90\x00"), "application/octet-stream")},
                headers=headers,
            )
            if r.status_code in (400, 415, 422):
                log(PASS, "POST /documents/upload (unsupported type → 4xx)", f"HTTP {r.status_code}")
            else:
                log(WARN, "POST /documents/upload (unsupported type)", f"Got {r.status_code} | {r.text[:100]}")
        except Exception as e:
            log(FAIL, "POST /documents/upload (unsupported type)", str(e))

        # ──────────────────────────────────────────────────────
        section("10. CHAT — SEND MESSAGE")
        # ──────────────────────────────────────────────────────
        try:
            r = client.post(
                f"{BASE}/chat",
                json={"message": "What is the Eiffel Tower?", "stream": False, "maxResults": 3},
                headers=headers,
                timeout=60,
            )
            body = r.json()
            if r.status_code == 200 and body.get("success"):
                conv_id = body["data"]["conversationId"]
                answer_preview = body["data"]["answer"][:80]
                sources = len(body["data"]["sources"])
                log(PASS, "POST /chat", f"conv_id={conv_id} | sources={sources} | answer='{answer_preview}...'")
            else:
                log(FAIL, "POST /chat", f"HTTP {r.status_code}", body)
        except Exception as e:
            log(FAIL, "POST /chat", str(e))

        # ──────────────────────────────────────────────────────
        section("11. CHAT — STREAM MESSAGE (FOLLOW-UP)")
        # ──────────────────────────────────────────────────────
        if conv_id:
            try:
                with client.stream(
                    "POST",
                    f"{BASE}/chat/stream",
                    json={"message": "Who built it?", "conversationId": conv_id, "stream": True},
                    headers=headers,
                    timeout=60,
                ) as r:
                    if r.status_code == 200:
                        events = []
                        for line in r.iter_lines():
                            if line.strip():
                                events.append(line)
                        log(PASS, "POST /chat/stream", f"Status 200 | received {len(events)} lines of stream data")
                    else:
                        log(FAIL, "POST /chat/stream", f"HTTP {r.status_code}")
            except Exception as e:
                log(FAIL, "POST /chat/stream", str(e))
        else:
            log(WARN, "POST /chat/stream", "skipped — no conv_id from first chat")

        # ──────────────────────────────────────────────────────
        section("12. CONVERSATIONS — LIST")
        # ──────────────────────────────────────────────────────
        try:
            r = client.get(f"{BASE}/chat/conversations", headers=headers)
            body = r.json()
            if r.status_code == 200 and body.get("success"):
                total = body["data"]["totalElements"]
                log(PASS, "GET /chat/conversations", f"totalElements={total}")
            else:
                log(FAIL, "GET /chat/conversations", f"HTTP {r.status_code}", body)
        except Exception as e:
            log(FAIL, "GET /chat/conversations", str(e))

        # ──────────────────────────────────────────────────────
        section("13. CONVERSATIONS — GET BY ID")
        # ──────────────────────────────────────────────────────
        if conv_id:
            try:
                r = client.get(f"{BASE}/chat/conversations/{conv_id}", headers=headers)
                body = r.json()
                if r.status_code == 200 and body.get("success"):
                    msgs = len(body["data"].get("messages") or [])
                    log(PASS, f"GET /chat/conversations/{{conv_id}}", f"messages={msgs}")
                else:
                    log(FAIL, f"GET /chat/conversations/{{conv_id}}", f"HTTP {r.status_code}", body)
            except Exception as e:
                log(FAIL, f"GET /chat/conversations/{{conv_id}}", str(e))
        else:
            log(WARN, "GET /chat/conversations/{conv_id}", "skipped — no conv_id")

        # ──────────────────────────────────────────────────────
        section("14. CLEANUP — DELETE CONVERSATION")
        # ──────────────────────────────────────────────────────
        if conv_id:
            try:
                r = client.delete(f"{BASE}/chat/conversations/{conv_id}", headers=headers)
                body = r.json()
                if r.status_code == 200 and body.get("success"):
                    log(PASS, f"DELETE /chat/conversations/{{conv_id}}", "deleted")
                else:
                    log(FAIL, f"DELETE /chat/conversations/{{conv_id}}", f"HTTP {r.status_code}", body)
            except Exception as e:
                log(FAIL, f"DELETE /chat/conversations/{{conv_id}}", str(e))

        # ──────────────────────────────────────────────────────
        section("15. CLEANUP — DELETE DOCUMENT")
        # ──────────────────────────────────────────────────────
        if doc_id:
            try:
                r = client.delete(f"{BASE}/documents/{doc_id}", headers=headers)
                body = r.json()
                if r.status_code == 200 and body.get("success"):
                    log(PASS, f"DELETE /documents/{{doc_id}}", "deleted")
                else:
                    log(FAIL, f"DELETE /documents/{{doc_id}}", f"HTTP {r.status_code}", body)
            except Exception as e:
                log(FAIL, f"DELETE /documents/{{doc_id}}", str(e))

    # ──────────────────────────────────────────────────────────
    section("SUMMARY")
    # ──────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    warned = sum(1 for r in results if r["status"] == "WARN")
    total = len(results)
    print(f"\n  Total : {total}")
    print(f"  \033[92mPassed\033[0m: {passed}")
    print(f"  \033[91mFailed\033[0m: {failed}")
    print(f"  \033[93mWarned\033[0m: {warned}")
    print()
    if failed:
        print("  \033[91mFailed tests:\033[0m")
        for r in results:
            if r["status"] == "FAIL":
                print(f"    - {r['test']}: {r['msg']}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run_tests()
