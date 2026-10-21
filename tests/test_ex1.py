"""Quick test for Exercise 1: PII shield."""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `import bot` works

print("=== Exercise 1 Test ===")
print("Loading bot module...")
from bot import mask_pii, mask_step, router, TICKETS

print("Module loaded OK\n")

# --- Test 1: Phone number masked ---
print("--- Test 1: Phone number masked ---")
text, count = mask_pii("Call me on 98450 12345 please")
print(f"  Output: '{text}'  Count: {count}")
assert "[PHONE]" in text and "98450" not in text and count == 1
print("  PASS\n")

# --- Test 2: Email masked ---
print("--- Test 2: Email masked ---")
text, count = mask_pii("Mail me at ravi.k@gmail.com")
print(f"  Output: '{text}'  Count: {count}")
assert "[EMAIL]" in text and "ravi.k@gmail.com" not in text
print("  PASS\n")

# --- Test 3: UPI masked ---
print("--- Test 3: UPI masked ---")
text, count = mask_pii("UPI id is ravi@okaxis")
print(f"  Output: '{text}'  Count: {count}")
assert "[UPI]" in text and "ravi@okaxis" not in text
print("  PASS\n")

# --- Test 4: Card number masked ---
print("--- Test 4: Card number masked ---")
text, count = mask_pii("Card 4111222233334444 was charged")
print(f"  Output: '{text}'  Count: {count}")
assert "[CARD]" in text and "4111222233334444" not in text
print("  PASS\n")

# --- Test 5: X1 – all PII types in one message ---
print("--- Test 5: X1 ticket – phone + email + UPI together ---")
x1 = "Please call me on 98450 12345 or mail ravi.k@gmail.com, UPI ravi@okaxis. FK1001 mein Britannia bread missing tha, refund do."
text, count = mask_pii(x1)
print(f"  Output: '{text}'")
print(f"  Count:  {count}")
assert "98450" not in text and "ravi.k@gmail.com" not in text and "ravi@okaxis" not in text
assert count >= 3
assert "FK1001" in text, "FAIL: order ID was incorrectly masked"
print("  PASS\n")

# --- Test 6: mask_step stashes original in _raw_input ---
print("--- Test 6: mask_step preserves _raw_input ---")
result = mask_step({"input": x1, "chat_history": []})
assert result["_raw_input"] == x1, "FAIL: _raw_input != original"
assert "98450" not in result["input"], "FAIL: phone not masked in input"
assert result["pii_masked_count"] >= 3
print(f"  _raw_input preserved: True  input masked: True  count: {result['pii_masked_count']}")
print("  PASS\n")

# --- Test 7: 18/18 routing regression ---
print("--- Test 7: 18/18 routing regression ---")
res = router.batch([{"input": t["message"], "chat_history": []} for t in TICKETS])
score = sum(r["route"] == t["expected_route"] for r, t in zip(res, TICKETS))
print(f"  Score: {score} / 18")
for r, t in zip(res, TICKETS):
    status = "PASS" if r["route"] == t["expected_route"] else "FAIL"
    print(f"    {status} {t['id']}: got={r['route']} expected={t['expected_route']}")
print()
print("DONE")
