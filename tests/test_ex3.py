"""Quick test for Exercise 3: retry + fallback."""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `import bot` works

print("=== Exercise 3 Test ===")
print("Loading bot module...")
from bot import router, TICKETS, resilient_llm, FALLBACK_REPLY, BACKUP_MODEL

print(f"BACKUP_MODEL: {BACKUP_MODEL}")
print(f"FALLBACK_REPLY: {FALLBACK_REPLY}")
print(f"resilient_llm type: {type(resilient_llm).__name__}")
print()

# Test 1: Single ticket with normal key
print("--- Test 1: Single ticket (T01) with normal key ---")
t = TICKETS[0]
r = router.invoke({"input": t["message"], "chat_history": []})
print(f"  {t['id']}: route={r['route']} expected={t['expected_route']} {'PASS' if r['route'] == t['expected_route'] else 'FAIL'}")
print()

# Test 2: Full 18-ticket regression
print("--- Test 2: 18-ticket regression ---")
res = router.batch([{"input": t["message"], "chat_history": []} for t in TICKETS])
score = sum(r["route"] == t["expected_route"] for r, t in zip(res, TICKETS))
print(f"Score: {score} / 18")
for r, t in zip(res, TICKETS):
    status = "PASS" if r["route"] == t["expected_route"] else "FAIL"
    print(f"  {status} {t['id']}: got={r['route']} expected={t['expected_route']}")
print()
print("DONE")
