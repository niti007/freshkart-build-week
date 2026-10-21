"""Quick test for Bonus: Reply guard."""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `import bot` works

print("=== Bonus: Reply Guard Test ===")
print("Loading bot module...")
from bot import (router, reply_chain, TICKETS, SAFE_REPLY, _GUARD_REGEX,
                 _check_reply, guarded_lane, LANES)

print("Module loaded OK\n")

# Helper
def run_ticket(tid):
    t = next(t for t in TICKETS if t["id"] == tid)
    decision = router.invoke({"input": t["message"], "chat_history": []})
    result = reply_chain.invoke(decision)
    return decision, result

# --- Test 1: Regex guard catches "₹780 refund today" ---
print("--- Test 1: Regex guard catches forbidden patterns ---")
assert _GUARD_REGEX.search("Your ₹780 will be refunded today"), "FAIL: regex missed ₹780 refund today"
assert _GUARD_REGEX.search("Rs 500 credited"), "FAIL: regex missed Rs 500"
assert _GUARD_REGEX.search("The route is human_review"), "FAIL: regex missed internal 'route'"
assert _GUARD_REGEX.search("internal note says"), "FAIL: regex missed 'internal note'"
assert not _GUARD_REGEX.search("A senior agent will contact you shortly"), "FAIL: false positive on safe text"
print("  PASS\n")

# --- Test 2: T09 (human_review) – reply should NOT contain ₹ amounts ---
print("--- Test 2: T09 (human_review, food safety) – no ₹ amounts in reply ---")
dec, res = run_ticket("T09")
reply = res["reply"]
print(f"  Route: {dec['route']}")
print(f"  Reply: {reply[:100]}...")
assert dec["route"] == "human_review", f"FAIL: route={dec['route']}"
# The reply should be clean (either the LLM was good, or the guard replaced it)
if _GUARD_REGEX.search(reply):
    print(f"  WARNING: guard regex found a match in the reply: {_GUARD_REGEX.search(reply).group()}")
    print("  This means the guard should have replaced it with SAFE_REPLY")
    assert reply == SAFE_REPLY, "FAIL: guard didn't replace a bad reply"
print("  PASS\n")

# --- Test 3: T01 (auto_refund) – NOT guarded, can mention amounts ---
print("--- Test 3: T01 (auto_refund) – NOT guarded, amounts are OK ---")
dec, res = run_ticket("T01")
reply = res["reply"]
print(f"  Route: {dec['route']}")
print(f"  Reply: {reply[:100]}...")
assert dec["route"] == "auto_refund", f"FAIL: route={dec['route']}"
# auto_refund lane is NOT guarded — it's fine to mention refund amounts here
print("  PASS (auto_refund lane not guarded)\n")

# --- Test 4: Direct test – force a bad draft through the guard ---
print("--- Test 4: Force bad draft through _check_reply ---")
bad = {"draft": "Don't worry, your ₹780 will be refunded today."}
verdict = _check_reply(bad)
print(f"  Draft: {bad['draft']}")
print(f"  Verdict: {verdict}")
assert verdict == "yes", f"FAIL: verdict should be 'yes' for bad draft, got '{verdict}'"
print("  PASS\n")

# --- Test 5: Direct test – clean draft passes guard ---
print("--- Test 5: Clean draft passes _check_reply ---")
good = {"draft": "I understand your concern. A senior agent will look into this and contact you soon."}
verdict = _check_reply(good)
print(f"  Draft: {good['draft']}")
print(f"  Verdict: {verdict}")
assert verdict.startswith("no"), f"FAIL: verdict should be 'no' for clean draft, got '{verdict}'"
print("  PASS\n")

# --- Test 6: 18/18 routing regression ---
print("--- Test 6: 18/18 routing regression ---")
results = router.batch([{"input": t["message"], "chat_history": []} for t in TICKETS])
score = sum(r["route"] == t["expected_route"] for r, t in zip(results, TICKETS))
print(f"  Score: {score} / 18")
for r, t in zip(results, TICKETS):
    status = "PASS" if r["route"] == t["expected_route"] else "FAIL"
    print(f"    {status} {t['id']}: got={r['route']} expected={t['expected_route']}")
assert score == 18, f"FAIL: {score}/18"
print("  PASS\n")

print("=== All Bonus tests passed ===")
