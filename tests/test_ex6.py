"""Quick test for Exercise 6: Model switcher at runtime."""
import sys, time
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `import bot` works

print("=== Exercise 6 Test ===")
print("Loading bot module...")
from bot import router, reply_chain, TICKETS, MODELS

print(f"Available models: {MODELS}")
print("Module loaded OK\n")

# --- Test 1: Same router, two models, both produce scores ---
print("--- Test 1: 18-ticket eval on each model ---")
for key, name in MODELS.items():
    cfg = {"configurable": {"model": key}}
    t0 = time.time()
    res = router.batch([{"input": t["message"], "chat_history": []} for t in TICKETS], config=cfg)
    elapsed = time.time() - t0
    score = sum(r["route"] == t["expected_route"] for r, t in zip(res, TICKETS))
    print(f"  {key} ({name}): {score}/18 in {elapsed:.1f}s")
    for r, t in zip(res, TICKETS):
        status = "PASS" if r["route"] == t["expected_route"] else "FAIL"
        print(f"    {status} {t['id']}: got={r['route']} expected={t['expected_route']}")
print("  PASS\n")

# --- Test 2: Switch model mid-conversation (single ticket) ---
print("--- Test 2: Same ticket, different models → both run without error ---")
t = next(t for t in TICKETS if t["id"] == "T01")
for key in MODELS:
    cfg = {"configurable": {"model": key}}
    dec = router.invoke({"input": t["message"], "chat_history": []}, config=cfg)
    res = reply_chain.invoke(dec, config=cfg)
    print(f"  {key}: route={dec['route']}, reply={res['reply'][:60]}...")
print("  PASS\n")

# --- Test 3: T18 (VIP Rs 2000) → both models give human_review ---
print("--- Test 3: T18 (VIP Rs 2000) → human_review with BOTH models ---")
t18 = next(t for t in TICKETS if t["id"] == "T18")
for key in MODELS:
    cfg = {"configurable": {"model": key}}
    dec = router.invoke({"input": t18["message"], "chat_history": []}, config=cfg)
    print(f"  {key}: route={dec['route']}, reasons={dec['reasons']}")
    assert dec["route"] == "human_review", f"FAIL: {key} gave route={dec['route']}"
print("  PASS (model choice cannot change this route — the Rs 1000 rule is in decide_route, not the LLM)\n")

print("=== All Exercise 6 tests passed ===")
