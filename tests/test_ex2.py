"""Quick test for Exercise 2: Language-aware reply."""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `import bot` works

print("=== Exercise 2 Test ===")
print("Loading bot module...")
from bot import router, reply_chain, TICKETS, parse_intent_and_language, LANGUAGES

print(f"Supported languages: {LANGUAGES}")
print("Module loaded OK\n")

# --- Test 1: parse_intent_and_language parses combined string ---
print("--- Test 1: parse_intent_and_language ---")
result = parse_intent_and_language("missing_item,angry,hindi")
print(f"  Input:  'missing_item,angry,hindi'")
print(f"  Output: {result}")
assert result["intent"] == "missing_item", f"FAIL: intent={result['intent']}"
assert result["language"] == "hindi", f"FAIL: language={result['language']}"
print("  PASS\n")

# --- Test 2: Fallback for bad parse ---
print("--- Test 2: Fallback for bad parse ---")
result = parse_intent_and_language("gibberish")
print(f"  Input:  'gibberish'")
print(f"  Output: {result}")
assert result["intent"] == "unclear", f"FAIL: intent={result['intent']}"
assert result["language"] == "english", f"FAIL: language={result['language']}"
print("  PASS\n")

# --- Test 3: X2 Kannada ticket – language detected ---
print("--- Test 3: X2 Kannada ticket → language detected ---")
x2 = "FK1007 ಆರ್ಡರ್ ತುಂಬಾ ತಡವಾಗಿ ಬಂತು, ಯಾಕೆ?"
dec = router.invoke({"input": x2, "chat_history": []})
lang = dec["signals"].get("language", "unknown")
print(f"  Message: {x2[:40]}...")
print(f"  Language: {lang}")
print(f"  Intent: {dec['signals']['intent']}")
print(f"  Route: {dec['route']}")
assert lang in LANGUAGES, f"FAIL: language '{lang}' not in {LANGUAGES}"
print("  PASS\n")

# --- Test 4: X3 Telugu ticket – language detected ---
print("--- Test 4: X3 Telugu ticket → language detected ---")
x3 = "FK1003 లో వచ్చిన పెరుగు ఎక్స్‌పైర్ అయిపోయింది, రీఫండ్ కావాలి"
dec = router.invoke({"input": x3, "chat_history": []})
lang = dec["signals"].get("language", "unknown")
print(f"  Message: {x3[:40]}...")
print(f"  Language: {lang}")
print(f"  Intent: {dec['signals']['intent']}")
print(f"  Route: {dec['route']}")
assert lang in LANGUAGES, f"FAIL: language '{lang}' not in {LANGUAGES}"
print("  PASS\n")

# --- Test 5: T01 ticket – language detected as valid ---
print("--- Test 5: T01 ticket → valid language detected ---")
t = next(t for t in TICKETS if t["id"] == "T01")
dec = router.invoke({"input": t["message"], "chat_history": []})
lang = dec["signals"].get("language", "unknown")
print(f"  Message: {t['message'][:50]}...")
print(f"  Language: {lang}")
assert lang in LANGUAGES, f"FAIL: language '{lang}' not in {LANGUAGES}"
print("  PASS\n")

# --- Test 6: Language field present in all 18 tickets ---
print("--- Test 6: Language field present in all 18 routed tickets ---")
res = router.batch([{"input": t["message"], "chat_history": []} for t in TICKETS])
for r, t in zip(res, TICKETS):
    lang = r["signals"].get("language")
    assert lang is not None, f"FAIL: {t['id']} has no language field"
    assert lang in LANGUAGES, f"FAIL: {t['id']} language='{lang}' not in {LANGUAGES}"
    print(f"    {t['id']}: language={lang}")
print("  PASS\n")

# --- Test 7: 18/18 routing regression ---
print("--- Test 7: 18/18 routing regression ---")
score = sum(r["route"] == t["expected_route"] for r, t in zip(res, TICKETS))
print(f"  Score: {score} / 18")
for r, t in zip(res, TICKETS):
    status = "PASS" if r["route"] == t["expected_route"] else "FAIL"
    print(f"    {status} {t['id']}: got={r['route']} expected={t['expected_route']}")
print()
print("DONE")
