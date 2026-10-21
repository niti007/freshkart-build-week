"""Quick test for Exercise 7: Live pipeline tracker."""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `import bot` works

print("=== Exercise 7 Test ===")
print("Loading bot module...")
from bot import router, tracked_router, TICKETS, TRACKER_NAMES

print(f"Tracked station names: {sorted(TRACKER_NAMES)}")
print("Module loaded OK\n")

# --- Test 1: T09 – all named stations appear with timings ---
print("--- Test 1: T09 (food poisoning) – all stations with timings ---")
t = next(t for t in TICKETS if t["id"] == "T09")
decision, events = tracked_router({"input": t["message"], "chat_history": []})
print(f"  Route: {decision['route']}")
print(f"  Events captured: {len(events)}")
seen_names = {e["name"] for e in events}
print(f"  Station names: {sorted(seen_names)}")

# Sort by start time and display
sorted_events = sorted(events, key=lambda e: e["start"])
for ev in sorted_events:
    elapsed = (ev["end"] - ev["start"]).total_seconds()
    print(f"    ✓ {ev['name']:25s} {elapsed:.2f}s")

# Check key stations are present
expected = {"intent", "order_lookup", "customer_lookup", "safety_keywords", "decide_route"}
missing = expected - seen_names
assert not missing, f"FAIL: missing stations: {missing}"
print("  PASS\n")

# --- Test 2: Sum of parts > wall time (proves parallelism) ---
print("--- Test 2: Sum of signal times > total wall time (parallel!) ---")
t_min = min(e["start"] for e in events)
t_max = max(e["end"] for e in events)
wall_time = (t_max - t_min).total_seconds()
sum_of_parts = sum((e["end"] - e["start"]).total_seconds() for e in events)
print(f"  Wall time:     {wall_time:.2f}s")
print(f"  Sum of parts:  {sum_of_parts:.2f}s")
print(f"  Ratio:         {sum_of_parts / wall_time:.1f}x")
assert sum_of_parts > wall_time, f"FAIL: sum ({sum_of_parts:.2f}s) should be > wall ({wall_time:.2f}s)"
print("  PASS (sum > wall proves parallel execution)\n")

# --- Test 3: Turn it off – router.invoke() still works without tracker ---
print("--- Test 3: router.invoke() works without tracker (tracker is optional) ---")
result = router.invoke({"input": t["message"], "chat_history": []})
print(f"  Route: {result['route']}")
assert result["route"] == "human_review", f"FAIL: route={result['route']}"
print("  PASS (pipeline works fine without tracker attached)\n")

# --- Test 4: 18/18 routing regression with tracked_router ---
print("--- Test 4: 18/18 routing regression via tracked_router ---")
score = 0
for t in TICKETS:
    dec, evts = tracked_router({"input": t["message"], "chat_history": []})
    ok = dec["route"] == t["expected_route"]
    score += ok
    status = "PASS" if ok else "FAIL"
    print(f"    {status} {t['id']}: got={dec['route']} expected={t['expected_route']} ({len(evts)} events)")
print(f"  Score: {score} / 18")
assert score == 18, f"FAIL: {score}/18"
print("  PASS\n")

print("=== All Exercise 7 tests passed ===")
