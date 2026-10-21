"""Quick test for Exercise 4: Ops hand-off card for escalations."""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `import bot` works

print("=== Exercise 4 Test ===")
print("Loading bot module...")
from bot import router, reply_chain, TICKETS, add_note

print("Module loaded OK\n")

# Helper: run router + reply_chain for a ticket
def run_ticket(tid):
    t = next(t for t in TICKETS if t["id"] == tid)
    decision = router.invoke({"input": t["message"], "chat_history": []})
    result = reply_chain.invoke(decision)
    return decision, result

# --- Test 1: T09 (food poisoning) → team=food_safety, priority=P1 ---
print("--- Test 1: T09 (food poisoning) → card with team=food_safety, priority=P1 ---")
dec, res = run_ticket("T09")
print(f"  Route: {dec['route']}")
print(f"  Reply: {res['reply'][:80]}...")
print(f"  Card:  {res['card']}")
card = res["card"]
assert dec["route"] == "human_review", f"FAIL: route={dec['route']}"
assert card is not None, "FAIL: card is None"
assert card["team"] == "food_safety", f"FAIL: team={card['team']}"
assert card["priority"] == "P1", f"FAIL: priority={card['priority']}"
print("  PASS\n")

# --- Test 2: T13 (double debit) → team=finance ---
print("--- Test 2: T13 (double debit) → card with team=finance ---")
dec, res = run_ticket("T13")
print(f"  Route: {dec['route']}")
print(f"  Card:  {res['card']}")
card = res["card"]
assert dec["route"] == "human_review", f"FAIL: route={dec['route']}"
assert card is not None, "FAIL: card is None"
assert card["team"] == "finance", f"FAIL: team={card['team']}"
print("  PASS\n")

# --- Test 3: T10 (rude rider) → team=rider_ops ---
print("--- Test 3: T10 (rude rider) → card with team=rider_ops ---")
dec, res = run_ticket("T10")
print(f"  Route: {dec['route']}")
print(f"  Card:  {res['card']}")
card = res["card"]
assert dec["route"] == "human_review", f"FAIL: route={dec['route']}"
assert card is not None, "FAIL: card is None"
assert card["team"] == "rider_ops", f"FAIL: team={card['team']}"
print("  PASS\n")

# --- Test 4: T01 (auto_refund) → card=None, no extra LLM call ---
print("--- Test 4: T01 (auto_refund) → card=None ---")
dec, res = run_ticket("T01")
print(f"  Route: {dec['route']}")
print(f"  Reply: {res['reply'][:80]}...")
print(f"  Card:  {res['card']}")
assert dec["route"] == "auto_refund", f"FAIL: route={dec['route']}"
assert res["card"] is None, f"FAIL: card is not None: {res['card']}"
print("  PASS\n")

# --- Test 5: 18/18 routing regression ---
print("--- Test 5: 18/18 routing regression ---")
results = router.batch([{"input": t["message"], "chat_history": []} for t in TICKETS])
score = sum(r["route"] == t["expected_route"] for r, t in zip(results, TICKETS))
print(f"  Score: {score} / 18")
for r, t in zip(results, TICKETS):
    status = "PASS" if r["route"] == t["expected_route"] else "FAIL"
    print(f"    {status} {t['id']}: got={r['route']} expected={t['expected_route']}")
assert score == 18, f"FAIL: {score}/18"
print("  PASS\n")

print("=== All Exercise 4 tests passed ===")
