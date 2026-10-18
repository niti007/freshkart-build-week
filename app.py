import os, uuid, time
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
                 get_order, get_customer, order_id_from, bulk_triage, compare_models)

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
        lang = m["signals"].get("language")
        if lang: st.markdown(f"🌐 **Language:** `{lang}`")
        model_used = m["signals"].get("model_used")
        if model_used:
            b_color = "green" if model_used == "main" else ("orange" if model_used == "backup" else "red")
            st.markdown(f":{b_color}[🤖 **Model:** `{model_used}`]")
        if m.get("pii_count", 0) > 0:
            st.markdown(f":violet[🛡️ **PII masked: {m['pii_count']} item(s)**]")
            st.caption("Prompt input sent to LLM:")
            st.code(m.get("masked_input", ""), language="text")
        if m.get("tracker"):
            st.markdown("⏱️ **Conveyor Belt Station Timings:**")
            for e in sorted(m["tracker"], key=lambda x: x["start_time"]):
                st.write(f"- ✓ `{e['name']}`: {e['duration']:.4f}s")
        st.json(m["signals"])

# Sidebar
sample = None
with st.sidebar:
    st.header("About")
    st.write("Rules-based support bot: signals in parallel, routing in plain Python, then a streamed reply.")
    selected_model = st.selectbox("LLM Model", ["mini", "llama"], index=0, help="Switch model at runtime via ConfigurableField")
    cfg = {"configurable": {"model": selected_model}}
    if st.button("Benchmark models (18 tickets)"):
        with st.spinner("Benchmarking mini vs llama on 18 tickets (max_concurrency=5)..."):
            cmp_res = compare_models()
        for m_name, res in cmp_res.items():
            st.write(f"**{m_name}**: {res['score']}/18 accurate ({res['time']:.2f}s)")
    st.divider()
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
            if m.get("card"):
                c = m["card"]
                st.info(f"📋 **Ops Escalation Card** [{c['priority']}] → **Team:** `{c['team']}`\n\n"
                        f"**Summary:** {c['summary_en']}\n\n**Next Step:** {c['next_step']}")
            if m["role"] == "assistant": why(m)
    if msg:
        with st.chat_message("user"):
            st.write(msg)
        history = get_history(ss.sid)
        tracker = []
        cfg_tracked = {"configurable": {"model": selected_model, "tracker": tracker}}
        with st.chat_message("assistant"):
            with st.status("Conveyor belt processing ticket...", expanded=True) as status_box:
                decision = router.invoke({"input": msg, "chat_history": history.messages}, config=cfg_tracked)
                for e in sorted(tracker, key=lambda x: x["start_time"]):
                    st.write(f"✓ **{e['name']}** `{e['duration']:.4f}s`")
                status_box.update(label=f"Belt routed: :{COLORS[decision['route']]}[**{decision['route']}**] ({len(tracker)} stations)", state="complete", expanded=False)
            card_holder = [None]
            def stream_tokens(gen):
                for chunk in gen:
                    if isinstance(chunk, dict):
                        if "card" in chunk and chunk["card"]: card_holder[0] = chunk["card"]
                        if "reply" in chunk and chunk["reply"]: yield chunk["reply"]
                    else:
                        yield chunk
            reply = st.write_stream(stream_tokens(reply_chain.stream(decision, config=cfg_tracked)))
            card_data = card_holder[0]
            if card_data:
                st.info(f"📋 **Ops Escalation Card** [{card_data['priority']}] → **Team:** `{card_data['team']}`\n\n"
                        f"**Summary:** {card_data['summary_en']}\n\n**Next Step:** {card_data['next_step']}")
            m = {"role": "assistant", "content": reply, "route": decision["route"],
                 "reasons": decision["reasons"], "signals": decision["signals"],
                 "pii_count": decision.get("pii_count", 0), "masked_input": decision.get("masked_input", ""),
                 "card": card_data, "tracker": list(tracker)}
            why(m)
        history.add_user_message(decision.get("masked_input", msg))
        history.add_ai_message(reply)
        ss.messages += [{"role": "user", "content": msg}, m]
        ss.log.append({"time": datetime.now().strftime("%H:%M:%S"), "order_id": decision["order_id"],
                       "intent": decision["signals"]["intent"], "language": decision["signals"].get("language", "english"),
                       "route": decision["route"], "team": card_data.get("team") if card_data else "-",
                       "priority": card_data.get("priority") if card_data else "-",
                       "summary": card_data.get("summary_en") if card_data else "-",
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

    st.divider()
    st.subheader("Bulk Triage (Morning Backlog)")
    uploaded_file = st.file_uploader("Upload tickets CSV (columns: id, message)", type=["csv"], key="bulk_csv_upload")
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file, dtype=str)
        st.caption(f"Loaded {len(df_upload)} rows from CSV:")
        st.dataframe(df_upload.head(3), use_container_width=True)
        if st.button("Triage", key="btn_run_bulk_triage"):
            with st.spinner(f"Triaging {len(df_upload)} tickets in parallel (concurrency=5)..."):
                t0 = time.time()
                triaged = bulk_triage.invoke(df_upload.to_dict("records"), config=cfg)
                elapsed = time.time() - t0
            df_triaged = pd.DataFrame(triaged)
            st.success(f"Triaged {len(df_triaged)} tickets in {elapsed:.2f}s!")
            route_counts = df_triaged["route"].value_counts().to_dict()
            rc_cols = st.columns(len(route_counts))
            for c, (rt, count) in zip(rc_cols, route_counts.items()):
                c.metric(rt, count)
            st.dataframe(df_triaged, use_container_width=True)

with tab_data:
    for name, df in [("orders", orders), ("customers", customers), ("inventory", inventory), ("riders", riders)]:
        st.subheader(name)
        st.dataframe(df, use_container_width=True)

with tab_how:
    st.code("""
 {input, chat_history}
        |
 RunnablePassthrough.assign(masked_input, pii_count)   regex: [PHONE], [EMAIL], [UPI], [CARD]
        |
 RunnableLambda(find_order_id)                         regex: FK1001 / fk1001 / ord_1001
        |
 RunnablePassthrough.assign(nlu)                       1 LLM call: intent, sentiment, language
        |
 RunnablePassthrough.assign(signals = RunnableParallel(
        intent | language | sentiment | order | customer | inventory | rider | safety | claimed_amount ))
        |
 RunnableLambda(decide_route)                          plain Python rules, no LLM
        |
 RunnableLambda(add_note)
        |
 RunnableParallel(reply = RunnableBranch(...), card = RunnableBranch(human_review -> card_chain, None))
        |
 Streamed reply + Ops escalation card                  (memory: RunnableWithMessageHistory)
""", language="text")
    st.table(pd.DataFrame([
        ("Cell 2 Setup", "get_llm, read", "Data tab"),
        ("Cell 9 PII Masking", "mask_pii", "Why this answer? (PII badge)"),
        ("Cell 10 RunnableLambda", "find_order_id", "Order info in sidebar"),
        ("Cell 12 Lookups", "get_order, get_customer, check_inventory, get_rider", "Why this answer? (signals)"),
        ("Cell 14 assign + memory", "order_id_from", "Order recovered from earlier chat turns"),
        ("Cell 16 Fault-Tolerant NLU", "primary_nlu, backup_nlu", "Why this answer? (intent, language, model)"),
        ("Cell 18 decide_route", "decide_route", "Route badge, Ops queue, metrics"),
        ("Cell 20 Robust Lanes", "lane, backup_llm, fallback", "Streamed reply in Chat with fallback"),
        ("Cell 21 Ops Card", "card_chain, card_branch", "Ops escalation card in Chat & Queue"),
        ("Cell 22 Memory", "full_chain, store, get_history", "New chat button"),
        ("Cell 23 Bulk Triage", "bulk_triage", "Ops queue bulk CSV triage"),
        ("Cell 24 Configurable Alternatives", "llm, compare_models", "Sidebar LLM Model switcher & Benchmark"),
        ("Cell 25 Live Pipeline Tracker", "track, with_listeners", "Conveyor Belt Status & timings in Chat"),
        ("Cell 26 Evaluation", "router", "Sample ticket + Send"),
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
