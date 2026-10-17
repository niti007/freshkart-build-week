import os, uuid
from datetime import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="FreshKart Support Desk", page_icon="🛒", layout="wide")
st.title("FreshKart Support Desk")
load_dotenv()
if not os.getenv("OPENROUTER_API_KEY"):
    st.error("OPENROUTER_API_KEY is missing. Copy .env.example to .env, add your key, and restart.")
    st.stop()
from bot import (TICKETS, orders, customers, inventory, riders, router, reply_chain, get_history,
                 get_order, get_customer, order_id_from)

ss = st.session_state
ss.setdefault("sid", uuid.uuid4().hex[:8])
ss.setdefault("messages", [])  # rendered chat: role, content, route, reasons, signals
ss.setdefault("log", [])       # one row per routed ticket (for the Ops tab)
COLORS = {"auto_refund": "green", "wismo": "blue", "substitution": "violet", "info": "gray",
          "policy_decline": "orange", "human_review": "red"}

def why(m):
    with st.expander("Why this answer?"):
        st.markdown(f"Route: :{COLORS[m['route']]}[**{m['route']}**]")
        st.markdown("Reasons: " + "; ".join(m["reasons"]))
        st.markdown(f"Language: **{m['signals'].get('language', 'english')}**")
        st.markdown(f"Intent answered by: `{m['signals'].get('intent_model', 'n/a')}`")
        if m.get("pii_count"):
            st.markdown(f":shield: PII masked: **{m['pii_count']} items** (before any prompt)")
        st.json(m["signals"])

# Sidebar
sample = None
with st.sidebar:
    st.header("About")
    st.write("Rules-based support bot: signals in parallel, routing in plain Python, then a streamed reply.")
    choice = st.selectbox("Try a sample ticket", TICKETS, format_func=lambda t: f"{t['id']}: {' '.join(t['message'].split()[:6])}...")
    if st.button("Send"):
        sample = choice["message"]
    if st.button("New chat"):
        ss.sid, ss.messages = uuid.uuid4().hex[:8], []
        st.rerun()

msg = st.chat_input("Write to FreshKart support...") or sample
tab_chat, tab_ops, tab_data, tab_how = st.tabs(["Chat", "Ops queue", "Data", "How it works"])

with tab_chat:
    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["role"] == "assistant": why(m)
    if msg:
        with st.chat_message("user"):
            st.write(msg)
        history = get_history(ss.sid)
        with st.chat_message("assistant"):
            with st.spinner("Checking order, customer and policy..."):
                decision = router.invoke({"input": msg, "chat_history": history.messages})
            reply = st.write_stream(reply_chain.stream(decision))
            m = {"role": "assistant", "content": reply, "route": decision["route"],
                 "reasons": decision["reasons"], "signals": decision["signals"],
                 "pii_count": decision["pii_count"]}
            why(m)
        history.add_user_message(msg)
        history.add_ai_message(reply)
        ss.messages += [{"role": "user", "content": msg}, m]
        ss.log.append({"time": datetime.now().strftime("%H:%M:%S"), "order_id": decision["order_id"],
                       "intent": decision["signals"]["intent"], "language": decision["signals"]["language"],
                       "route": decision["route"],
                       "reasons": "; ".join(decision["reasons"]), "message": msg})

with tab_ops:
    cols = st.columns(len(COLORS))
    for col, route in zip(cols, COLORS):
        col.metric(route, sum(r["route"] == route for r in ss.log))
    queue = [r for r in ss.log if r["route"] == "human_review"]
    st.subheader("Human review queue")
    if queue:
        st.dataframe(pd.DataFrame(queue).drop(columns="route"), use_container_width=True)
    else:
        st.info("No tickets sent to a human yet in this session.")

with tab_data:
    for name, df in [("orders", orders), ("customers", customers), ("inventory", inventory), ("riders", riders)]:
        st.subheader(name)
        st.dataframe(df, use_container_width=True)

with tab_how:
    st.code("""
 {input, chat_history}
        |
 RunnablePassthrough.assign(masked_input, pii_count)   Ex1: mask phone/email/UPI/card for prompts
        |
 RunnableLambda(find_order_id)        regex: FK1001 / fk1001 / ord_1001
        |
 RunnablePassthrough.assign(signals = RunnableParallel(
        intent (LLM) | language (LLM, Ex2) | order | customer | inventory | rider | safety | claimed_amount ))
        |
 RunnableLambda(unpack_intent)        Ex3: intent + which model answered (retry -> backup model -> "unclear")
        |
 RunnableLambda(decide_route)         plain Python rules, no LLM
        |
 RunnableBranch: auto_refund | wismo | substitution | info | policy_decline | human_review
        |
 StrOutputParser -> streamed reply in the customer's language   (memory: RunnableWithMessageHistory)
""", language="text")
    st.table(pd.DataFrame([
        ("Cell 2 Setup", "get_llm, read", "Data tab"),
        ("Cell 10 RunnableLambda", "find_order_id", "Order info in sidebar"),
        ("Cell 12 Lookups", "get_order, get_customer, check_inventory, get_rider", "Why this answer? (signals)"),
        ("Cell 14 assign + memory", "order_id_from", "Order recovered from earlier chat turns"),
        ("Cell 16 RunnableParallel", "signals, parse_intent", "Why this answer? (signals)"),
        ("Cell 18 decide_route", "decide_route", "Route badge, Ops queue, metrics"),
        ("Cell 20 RunnableBranch", "add_note, branch, reply_chain", "Streamed reply in Chat"),
        ("Cell 22 Memory", "full_chain, store, get_history", "New chat button"),
        ("Cell 24 Evaluation", "router", "Sample ticket + Send"),
    ], columns=["Notebook cell", "bot.py", "App widget"]))

# Sidebar order panel for the latest message
with st.sidebar:
    if ss.messages:
        last = [m for m in ss.messages if m["role"] == "user"][-1]["content"]
        oid = order_id_from({"input": last, "chat_history": get_history(ss.sid).messages})
        if oid and get_order(oid):
            o, c = get_order(oid), get_customer(oid)
            st.divider()
            st.caption(f"{oid}: {o['status']}, Rs {o['amount']}, {o['payment_mode']}")
            st.caption(f"{c['name']} ({c['tier']}), refunds 30d: {c['refunds_30d']}")
