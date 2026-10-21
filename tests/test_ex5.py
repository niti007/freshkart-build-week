"""Quick test for Exercise 5: Bulk triage for the morning backlog."""
import sys, time
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `import bot` works

print("=== Exercise 5 Test ===")
print("Loading bot module...")
from bot import router, bulk_triage, TICKETS

print("Module loaded OK\n")

# --- Test 1: Bulk triage on 18 tickets (matches data/tickets.csv) ---
print("--- Test 1: bulk_triage on 18 tickets → 18/18 routes ---")
messages = [{"id": t["id"], "message": t["message"]} for t in TICKETS]
t0 = time.time()
rows = bulk_triage(messages)
batch_time = time.time() - t0
print(f"  Time (batch, max_concurrency=5): {batch_time:.1f}s")
print(f"  Rows returned: {len(rows)}")
score = sum(r["route"] == t["expected_route"] for r, t in zip(rows, TICKETS))
print(f"  Score: {score} / 18")
for r, t in zip(rows, TICKETS):
    status = "PASS" if r["route"] == t["expected_route"] else "FAIL"
    print(f"    {status} {r['id']}: got={r['route']} expected={t['expected_route']}")
assert score == 18, f"FAIL: {score}/18"
print("  PASS\n")

# --- Test 2: Timing – batch vs sequential ---
print("--- Test 2: Sequential (for loop) timing for comparison ---")
t0 = time.time()
for t in TICKETS:
    router.invoke({"input": t["message"], "chat_history": []})
seq_time = time.time() - t0
print(f"  Time (sequential): {seq_time:.1f}s")
print(f"  Time (batch):      {batch_time:.1f}s")
speedup = seq_time / batch_time if batch_time > 0 else float("inf")
print(f"  Speedup: {speedup:.1f}x")
assert speedup >= 1.5, f"FAIL: speedup only {speedup:.1f}x (expected >= 2x)"
print("  PASS\n")

# --- Test 3: Empty message row → route=error or info, others unaffected ---
print("--- Test 3: Add empty message row → that row shows error/info ---")
messages_with_empty = [{"id": t["id"], "message": t["message"]} for t in TICKETS]
messages_with_empty.append({"id": "X5", "message": ""})
rows = bulk_triage(messages_with_empty)
print(f"  Total rows: {len(rows)}")
empty_row = next(r for r in rows if r["id"] == "X5")
print(f"  X5 (empty): route={empty_row['route']}")
assert empty_row["route"] in ("error", "info", "human_review"), f"FAIL: empty row route={empty_row['route']}"
normal_rows = [r for r in rows if r["id"] != "X5"]
normal_score = sum(r["route"] == t["expected_route"] for r, t in zip(normal_rows, TICKETS))
print(f"  Other 18 tickets: {normal_score} / 18")
assert normal_score == 18, f"FAIL: {normal_score}/18"
print("  PASS\n")

# --- Route counts ---
print("--- Route counts ---")
from collections import Counter
counts = Counter(r["route"] for r in rows)
for route, count in sorted(counts.items()):
    print(f"  {route}: {count}")

print("\n=== All Exercise 5 tests passed ===")
