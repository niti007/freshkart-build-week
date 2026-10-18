import os, sys, time, pytest
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
from bot import (
    router, reply_chain, TICKETS, mask_pii, compare_models,
    bulk_triage, check_reply_safety, SAFE_TEMPLATES
)

# Extra test cases defined in EXERCISES.md
EXTRA_TICKETS = [
    {
        "id": "X1",
        "message": "Please call me on 98450 12345 or mail ravi.k@gmail.com, UPI ravi@okaxis. FK1001 mein Britannia bread missing tha, refund do.",
        "expected_route": "auto_refund",
        "order_id": "FK1001",
        "test_feature": "Ex 1: PII Shield"
    },
    {
        "id": "X2",
        "message": "FK1007 ಆರ್ಡರ್ ತುಂಬಾ ತಡವಾಗಿ ಬಂತು, ಯಾಕೆ?",
        "expected_route": "wismo",
        "order_id": "FK1007",
        "test_feature": "Ex 2: Kannada Language Signal"
    },
    {
        "id": "X3",
        "message": "FK1003 లో వచ్చిన పెరుగు ఎక్స్‌పైర్ అయిపోయింది, రీఫండ్ కావాలి",
        "expected_route": "auto_refund",
        "order_id": "FK1003",
        "test_feature": "Ex 2: Telugu Language Signal"
    },
    {
        "id": "X4",
        "message": "FK1009 ka khana kha ke bacche ko ulti ho rahi hai",
        "expected_route": "human_review",
        "order_id": "FK1009",
        "test_feature": "Ex 4: P1 Food Safety Card"
    },
]

HAS_KEY = bool(os.getenv("OPENROUTER_API_KEY"))

@pytest.mark.skipif(not HAS_KEY, reason="No OPENROUTER_API_KEY")
def test_all_18_standard_tickets():
    """Validates routing on all 18 standard tickets from tickets.csv."""
    results = router.batch([{"input": t["message"], "chat_history": []} for t in TICKETS],
                           config={"max_concurrency": 5})
    wrong = [t["id"] for t, r in zip(TICKETS, results) if r["route"] != t["expected_route"]]
    assert len(wrong) == 0, f"Mismatched routes on: {wrong}"

@pytest.mark.skipif(not HAS_KEY, reason="No OPENROUTER_API_KEY")
def test_extra_ticket_x1_pii_masking():
    """X1: Validates phone, email, and UPI masking before NLU."""
    t = EXTRA_TICKETS[0]
    masked, count = mask_pii(t["message"])
    assert count >= 3, f"Expected at least 3 PII items masked, found {count}"
    assert "[PHONE]" in masked
    assert "[EMAIL]" in masked
    assert "[UPI]" in masked
    res = router.invoke({"input": t["message"], "chat_history": []})
    assert res["order_id"] == "FK1001"
    assert res["route"] == "auto_refund"

@pytest.mark.skipif(not HAS_KEY, reason="No OPENROUTER_API_KEY")
def test_extra_ticket_x2_kannada_language():
    """X2: Validates Kannada NLU detection & wismo routing."""
    t = EXTRA_TICKETS[1]
    res = router.invoke({"input": t["message"], "chat_history": []})
    assert res["order_id"] == "FK1007"
    assert res["signals"]["language"] == "kannada"
    assert res["route"] == "wismo"

@pytest.mark.skipif(not HAS_KEY, reason="No OPENROUTER_API_KEY")
def test_extra_ticket_x3_telugu_language():
    """X3: Validates Telugu NLU detection & policy evaluation."""
    t = EXTRA_TICKETS[2]
    res = router.invoke({"input": t["message"], "chat_history": []})
    assert res["order_id"] == "FK1003"
    assert res["signals"]["language"] == "telugu"
    assert res["route"] == "auto_refund"

@pytest.mark.skipif(not HAS_KEY, reason="No OPENROUTER_API_KEY")
def test_extra_ticket_x4_food_safety_escalation_card():
    """X4: Validates P1 food_safety Ops escalation card generation."""
    t = EXTRA_TICKETS[3]
    res = router.invoke({"input": t["message"], "chat_history": []})
    assert res["route"] == "human_review"
    reply_out = reply_chain.invoke(res)
    card = reply_out.get("card")
    assert card is not None, "Expected escalation card for X4"
    assert card["team"] == "food_safety"
    assert card["priority"] == "P1"

@pytest.mark.skipif(not HAS_KEY, reason="No OPENROUTER_API_KEY")
def test_exercise_5_bulk_triage_resilience():
    """Validates bulk triage concurrency and empty row resilience."""
    inputs = [{"id": "ERR1", "message": ""}, TICKETS[0]]
    triaged = bulk_triage.invoke(inputs)
    assert len(triaged) == 2
    assert triaged[0]["id"] == "ERR1"
    assert triaged[1]["route"] == TICKETS[0]["expected_route"]

@pytest.mark.skipif(not HAS_KEY, reason="No OPENROUTER_API_KEY")
def test_exercise_6_model_switch_and_t18_invariant():
    """Validates dynamic model switching and VIP T18 amount>1000 invariant on both models."""
    t18 = next(t for t in TICKETS if t["id"] == "T18")
    res_mini = router.invoke({"input": t18["message"], "chat_history": []}, config={"configurable": {"model": "mini"}})
    res_llama = router.invoke({"input": t18["message"], "chat_history": []}, config={"configurable": {"model": "llama"}})
    assert res_mini["route"] == "human_review"
    assert res_llama["route"] == "human_review"
    assert "amount>1000" in res_mini["reasons"]
    assert "amount>1000" in res_llama["reasons"]

@pytest.mark.skipif(not HAS_KEY, reason="No OPENROUTER_API_KEY")
def test_exercise_7_live_pipeline_tracker_and_overlap():
    """Validates station tracking and parallel overlap savings."""
    tracker = []
    t09 = next(t for t in TICKETS if t["id"] == "T09")
    t0 = time.time()
    res = router.invoke({"input": t09["message"], "chat_history": []}, config={"configurable": {"tracker": tracker}})
    router_time = time.time() - t0
    
    reply_tracker = []
    t0 = time.time()
    reply_out = reply_chain.invoke(res, config={"configurable": {"tracker": reply_tracker}})
    reply_time = time.time() - t0

    tracked_names = {e["name"] for e in tracker + reply_tracker}
    assert "order_lookup" in tracked_names
    assert "intent_llm" in tracked_names
    assert "reply_generation" in tracked_names
    assert "card_generation" in tracked_names

    reply_e = next(e for e in reply_tracker if e["name"] == "reply_generation")
    card_e = next(e for e in reply_tracker if e["name"] == "card_generation")
    assert reply_e["duration"] + card_e["duration"] > reply_time * 0.8

@pytest.mark.skipif(not HAS_KEY, reason="No OPENROUTER_API_KEY")
def test_bonus_reply_guard_interception():
    """Validates that unsafe refund promises are intercepted and replaced."""
    unsafe = {"draft": "Don't worry, your ₹780 will be refunded today."}
    safe = {"draft": "A senior support specialist is reviewing your case."}
    assert check_reply_safety(unsafe)["verdict"] == "unsafe"
    assert check_reply_safety(safe)["verdict"] == "safe"
