"""Interfaz Streamlit de NutriGuía (reutiliza el agente de rag_agent.py)."""
import os
import uuid

import streamlit as st

# En Streamlit Cloud la clave va en "Secrets"; en local, en .env
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:  # noqa: BLE001  (no hay secrets.toml en local)
    pass

from rag_agent import AGENT_NAME, DOMAIN_NAME, build_agent, build_vectorstore, invoke_agent

st.set_page_config(page_title=AGENT_NAME, page_icon="🥗")
st.title(f"🥗 {AGENT_NAME}")
st.caption(f"Asistente experto en {DOMAIN_NAME} · RAG condicional con Gemini + ChromaDB + LangGraph")


@st.cache_resource(show_spinner="Cargando base de conocimiento (la primera vez puede tardar unos minutos)...")
def get_agent():
    return build_agent(build_vectorstore())


if not os.getenv("GOOGLE_API_KEY"):
    st.error("Falta GOOGLE_API_KEY (archivo .env o Secrets de Streamlit).")
    st.stop()

try:
    agent = get_agent()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo iniciar el agente: {type(e).__name__}: {str(e)[:400]}")
    st.info("Si es un error de cuota (429), espera unos minutos y recarga la página: "
            "la indexación continúa donde se quedó.")
    st.stop()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []

with st.sidebar:
    st.header("Conversación")
    if st.button("🗑️ Nueva conversación"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.caption("Cada conversación tiene su propia memoria (thread_id).")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

if question := st.chat_input("Pregunta algo sobre nutrición..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            r = invoke_agent(agent, question, st.session_state.thread_id)
        st.markdown(r["answer"])
        meta = ("🔎 RAG · fuentes: " + (", ".join(r["sources"]) or "ninguna relevante")
                if r["use_rag"] else "💬 Respuesta directa (sin consultar la base)")
        st.caption(meta)
    st.session_state.history.append({"role": "assistant", "content": r["answer"], "meta": meta})
