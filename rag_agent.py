"""
NutriGuía — Agente experto con RAG condicional (Gemini + ChromaDB + LangGraph).

Este módulo es la ÚNICA fuente del código del agente: el notebook y la app de
Streamlit lo reutilizan. Las secciones marcadas con "# ====== SECTION: ..."
son las que el notebook muestra celda a celda.

Para cambiar de dominio: edita DOMAIN_* en la sección config, SYSTEM_PROMPT y
ROUTER_PROMPT en la sección prompts, y cambia los documentos de ./docs.
"""

# ====== SECTION: config
import os
import re
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()  # lee GOOGLE_API_KEY de .env si existe (nunca va en el código)

# --- Dominio -----------------------------------------------------------------
DOMAIN_NAME = "nutrición y alimentación saludable"
AGENT_NAME = "NutriGuía"

# --- Modelos (Google los renueva a menudo: si ves un 404, mira ai.google.dev/gemini-api/docs/models) ---
LLM_MODEL = "gemini-3.5-flash-lite"          # ligero: suele tener más cuota en el plan gratuito
LLM_FALLBACK_MODEL = "gemini-3.1-flash-lite"  # respaldo si el principal falla (cuota, saturación...)
# Con facturación activa puedes subir a "gemini-3.8-flash" como LLM_MODEL para respuestas más potentes.
EMBEDDING_MODEL = "models/gemini-embedding-001"

# --- Base vectorial ----------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent   # junto a este archivo (local, Streamlit Cloud)
except NameError:                                # notebook: no existe __file__
    BASE_DIR = Path.cwd()
DOCS_DIR = str(BASE_DIR / "docs")
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "conocimiento_experto"
CHUNK_SIZE = 1000        # caracteres por fragmento
CHUNK_OVERLAP = 150      # solapamiento para no cortar ideas por la mitad
BATCH_SIZE = 20          # fragmentos por llamada de embeddings (cuotas del plan gratuito)

# --- Recuperación ------------------------------------------------------------
TOP_K = 4                # fragmentos que se recuperan
MIN_RELEVANCE = 0.35     # similitud coseno mínima (calíbrala en la celda de verificación)

# --- Memoria -----------------------------------------------------------------
MAX_HISTORY_MESSAGES = 10  # mensajes recientes que se envían al LLM


# ====== SECTION: indexing
def clean_text(text: str) -> str:
    """Limpieza ligera: une palabras cortadas por guion, normaliza espacios y saltos."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(docs_dir: str = DOCS_DIR) -> List[Document]:
    """Carga PDFs, CSVs, TXT y MD de la carpeta de documentos."""
    docs: List[Document] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".csv":
            loader = CSVLoader(str(path), encoding="utf-8")
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            continue
        loaded = loader.load()
        for d in loaded:
            d.metadata["source_name"] = path.name
            d.page_content = clean_text(d.page_content)
        loaded = [d for d in loaded if len(d.page_content) > 50]  # elimina ruido/páginas vacías
        docs.extend(loaded)
        print(f"  · {path.name}: {len(loaded)} páginas/filas útiles")
    if not docs:
        found = sorted(p.name for p in Path(docs_dir).rglob("*")) if Path(docs_dir).exists() else []
        raise FileNotFoundError(
            f"No hay documentos legibles en '{docs_dir}' (existe: {Path(docs_dir).exists()}; "
            f"contenido: {found or 'vacío'}). Añade al menos 3 PDFs/CSVs/TXT (~20 páginas) "
            "a la carpeta docs/ del repositorio (no basta con tenerlos en tu ordenador)."
        )
    return docs


def _new_chroma() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # similitud coseno (scores 0-1)
    )


def build_vectorstore(rebuild: bool = False) -> Chroma:
    """Devuelve la colección de Chroma indexando solo los fragmentos que falten.
    Es reanudable: si una ejecución se corta (cuota, red), la siguiente continúa donde se quedó."""
    vs = _new_chroma()
    if rebuild and vs._collection.count() > 0:
        vs.delete_collection()
        vs = _new_chroma()

    print("Cargando documentos...")
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    ids = [f"chunk-{i}" for i in range(len(chunks))]  # ids estables -> permite reanudar

    existing = set(vs.get(include=[])["ids"])
    pending = [(i, c) for i, c in zip(ids, chunks) if i not in existing]
    if not pending:
        print(f"Colección completa: {len(chunks)} fragmentos (se reutiliza).")
        return vs

    print(f"{len(chunks)} fragmentos en total; faltan {len(pending)} por indexar. Generando embeddings...")
    last_error = None
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        for attempt in range(5):  # reintentos por si salta el límite de cuota
            try:
                vs.add_documents([c for _, c in batch], ids=[i for i, _ in batch])
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                wait = 20 * (attempt + 1)
                print(f"  Error ({type(e).__name__}: {str(e)[:150]}); reintento en {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError(
                "No se pudieron generar embeddings tras varios reintentos. "
                f"Último error: {type(last_error).__name__}: {str(last_error)[:300]}"
            ) from last_error
        time.sleep(2)
        print(f"  {min(start + BATCH_SIZE, len(pending))}/{len(pending)}")
    print(f"Indexados {vs._collection.count()} fragmentos en '{CHROMA_DIR}'.")
    return vs


# ====== SECTION: prompts
SYSTEM_PROMPT = f"""Eres {AGENT_NAME}, un asistente experto en {DOMAIN_NAME}.

ROL
- Orientas a personas adultas sin formación técnica con información rigurosa y fácil de entender.
- No eres médico ni dietista: no diagnosticas ni prescribes dietas personalizadas ni tratamientos.

TONO
- Cercano, claro y breve. Responde en el idioma del usuario (por defecto, español).
- Explica los términos técnicos la primera vez que los uses. Usa listas solo si ayudan.
- Saluda solo en el primer mensaje de la conversación; en los siguientes, ve directo a la respuesta.

USO DE LA INFORMACIÓN
- Cuando recibas un bloque «CONTEXTO RECUPERADO», basa tu respuesta en él y no inventes datos,
  cifras ni fuentes que no aparezcan ahí. Cita el documento entre corchetes, p. ej. [guia.pdf, p.12].
- Si el contexto no cubre la pregunta, dilo con honestidad («No encuentro esa información en mis
  documentos») y, como mucho, ofrece orientación general señalando claramente que no procede de ellos.
- Si no recibes contexto, la pregunta es conversacional o trata sobre la propia conversación:
  responde con naturalidad usando el historial.

LÍMITES
- Si el usuario describe síntomas, enfermedades, embarazo o trastornos de la conducta alimentaria,
  da información general y recomienda consultar a un profesional sanitario.
- Si la pregunta queda fuera de {DOMAIN_NAME}, indica amablemente que tu especialidad es otra y
  reconduce la conversación.
"""

ROUTER_PROMPT = f"""Eres el enrutador de un asistente experto en {DOMAIN_NAME}.
Decide si, para responder al ÚLTIMO mensaje del usuario, hace falta consultar la base de
conocimiento (documentos especializados).

use_rag = true cuando:
- Pide datos, recomendaciones, cantidades, comparaciones o explicaciones técnicas del dominio.
- Es una pregunta de seguimiento que necesita información nueva del dominio
  (p. ej. «¿y si me paso de esa cantidad?»).

use_rag = false cuando:
- Saludos, agradecimientos, despedidas o charla social.
- Preguntas sobre la propia conversación o sobre ti («¿qué te pregunté antes?», «¿quién eres?»).
- Peticiones de reformular, resumir o simplificar la respuesta anterior sin datos nuevos.
- Peticiones totalmente ajenas al dominio (se responderá reconduciendo la conversación).

Además redacta `standalone_query`: la pregunta reescrita para que se entienda SIN el historial
(resuelve pronombres y referencias como «eso», «y en niños?»). Si use_rag es false, repite la
pregunta tal cual."""


# ====== SECTION: agent
class RouteDecision(BaseModel):
    """Decisión del enrutador sobre si hay que consultar la base vectorial."""

    use_rag: bool = Field(description="True si hay que consultar la base de conocimiento")
    standalone_query: str = Field(description="Pregunta autocontenida para buscar en la base")
    reason: str = Field(description="Justificación breve de la decisión")


class AgentState(MessagesState):
    """Estado del grafo: `messages` (historial) + campos de trabajo de cada turno."""

    use_rag: bool
    standalone_query: str
    route_reason: str
    context: str
    sources: List[str]


def _text(content) -> str:
    """Extrae texto de un mensaje (Gemini puede devolver str o lista de partes)."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def _format_history(messages, n: int = 6) -> str:
    lines = []
    for m in messages[-n:]:
        role = "Usuario" if m.type == "human" else "Asistente"
        lines.append(f"{role}: {_text(m.content)[:600]}")
    return "\n".join(lines)


def _invoke_with_retry(runnable, payload, tries: int = 4):
    """Reintenta con espera creciente SOLO ante saturación temporal (503).
    Un 429 (cuota agotada) o un 404 (modelo inexistente) no se arreglan esperando:
    se propagan para que el llamador use el modelo de respaldo."""
    for i in range(tries):
        try:
            return runnable.invoke(payload)
        except Exception as e:  # noqa: BLE001
            temporal = "503" in str(e) or "UNAVAILABLE" in str(e)
            if temporal and i < tries - 1:
                wait = 3 * 2**i  # 3, 6, 12 s
                print(f"  ⏳ Gemini saturado; reintento {i + 1}/{tries - 1} en {wait}s...")
                time.sleep(wait)
                continue
            raise


def _with_fallback(primary, fallback, payload):
    """Prueba el modelo principal y, si falla, el de respaldo."""
    try:
        return _invoke_with_retry(primary, payload)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ Modelo principal no disponible ({type(e).__name__}); uso el de respaldo.")
        return _invoke_with_retry(fallback, payload)


def build_agent(vectorstore: Chroma):
    """Construye y compila el grafo: router -> (retrieve) -> generate, con memoria."""
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    llm_fallback = ChatGoogleGenerativeAI(model=LLM_FALLBACK_MODEL)
    router_llm = llm.with_structured_output(RouteDecision)
    router_fallback = llm_fallback.with_structured_output(RouteDecision)

    # Nodo 1: decide si se usa RAG (la "condición" del RAG condicional)
    def route_question(state: AgentState) -> dict:
        question = _text(state["messages"][-1].content)
        prompt = (
            f"{ROUTER_PROMPT}\n\nHISTORIAL RECIENTE:\n{_format_history(state['messages'][:-1])}"
            f"\n\nÚLTIMO MENSAJE DEL USUARIO:\n{question}"
        )
        try:
            d: RouteDecision = _with_fallback(router_llm, router_fallback, prompt)
            return {"use_rag": d.use_rag, "standalone_query": d.standalone_query or question,
                    "route_reason": d.reason}
        except Exception as e:  # noqa: BLE001  -> ante la duda, consultar la base
            return {"use_rag": True, "standalone_query": question,
                    "route_reason": f"fallback por error del router ({type(e).__name__})"}

    # Nodo 2 (solo si use_rag): recupera fragmentos relevantes de Chroma
    def retrieve(state: AgentState) -> dict:
        results = vectorstore.similarity_search_with_relevance_scores(
            state["standalone_query"], k=TOP_K
        )
        good = [(d, s) for d, s in results if s >= MIN_RELEVANCE]
        blocks, sources = [], []
        for d, s in good:
            name = d.metadata.get("source_name", "documento")
            page = d.metadata.get("page")
            tag = f"{name}, p.{page + 1}" if isinstance(page, int) else name
            blocks.append(f"[{tag}] (relevancia {s:.2f})\n{d.page_content}")
            if tag not in sources:
                sources.append(tag)
        return {"context": "\n\n---\n\n".join(blocks), "sources": sources}

    # Nodo 3: genera la respuesta con el system prompt + (opcional) contexto + memoria
    def generate(state: AgentState) -> dict:
        system = SYSTEM_PROMPT
        if state.get("use_rag"):
            if state.get("context"):
                system += f"\n\nCONTEXTO RECUPERADO:\n{state['context']}"
            else:
                system += ("\n\nSe consultó la base de conocimiento pero NO se encontró información "
                           "relevante. Indícalo al usuario siguiendo las reglas de uso de la información.")
        history = trim_messages(
            state["messages"], strategy="last", token_counter=len,
            max_tokens=MAX_HISTORY_MESSAGES, start_on="human", include_system=False,
        )
        prompt_msgs = [SystemMessage(content=system)] + history
        response = _with_fallback(llm, llm_fallback, prompt_msgs)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("route_question", route_question)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.add_edge(START, "route_question")
    builder.add_conditional_edges(
        "route_question",
        lambda s: "retrieve" if s["use_rag"] else "generate",
        {"retrieve": "retrieve", "generate": "generate"},
    )
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    # MemorySaver guarda el estado por thread_id -> memoria conversacional
    return builder.compile(checkpointer=MemorySaver())


def check_models() -> None:
    """Comprueba que los modelos configurados responden (así un 404 se detecta antes de chatear)."""
    for name in (LLM_MODEL, LLM_FALLBACK_MODEL):
        try:
            _invoke_with_retry(ChatGoogleGenerativeAI(model=name), "Responde solo: ok")
            print(f"✅ {name} disponible")
        except Exception as e:  # noqa: BLE001
            print(f"❌ {name}: {type(e).__name__}: {str(e)[:200]}")


def invoke_agent(agent, question: str, thread_id: str = "default") -> dict:
    """Envía una pregunta al agente y devuelve respuesta + metadatos de la ruta."""
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)], "context": "", "sources": []},
        config={"configurable": {"thread_id": thread_id}},
    )
    return {
        "answer": _text(result["messages"][-1].content),
        "use_rag": result.get("use_rag", False),
        "query": result.get("standalone_query", ""),
        "reason": result.get("route_reason", ""),
        "sources": result.get("sources", []),
    }


def ask(agent, question: str, thread_id: str = "default") -> dict:
    """Versión para notebook: invoca al agente y lo muestra formateado."""
    r = invoke_agent(agent, question, thread_id)
    print(f"🧑 {question}")
    route = "RAG (consulta la base)" if r["use_rag"] else "DIRECTO (sin consultar la base)"
    print(f"🔀 Ruta: {route} — {r['reason']}")
    if r["use_rag"]:
        print(f"🔎 Consulta: {r['query']}")
        print(f"📚 Fuentes: {', '.join(r['sources']) if r['sources'] else 'ninguna (bajo umbral de relevancia)'}")
    print(f"🤖 {r['answer']}\n")
    return r


def chat_loop(agent, thread_id: str = "chat-interactivo") -> None:
    """Chat interactivo en el notebook. Escribe 'salir' para terminar."""
    print(f"Chat con {AGENT_NAME}. Escribe 'salir' para terminar.\n")
    while True:
        q = input("Tú: ").strip()
        if q.lower() in {"salir", "exit", "quit", ""}:
            print("Hasta pronto 👋")
            break
        ask(agent, q, thread_id)
