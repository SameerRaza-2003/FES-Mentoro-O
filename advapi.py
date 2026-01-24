# api.py
import os
import re
import time
import asyncio
import logging
from typing import Generator, Iterable, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from openai import OpenAI
from pinecone import Pinecone

# ---------------------------------
# 🔧 Setup
# ---------------------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
INDEX_NAME = os.getenv("PINECONE_INDEX", "fes-embeddings-data")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "__default__")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in environment/.env")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing in environment/.env")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

app = FastAPI(title="FES Chatbot API (RAG + SSE)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fes-website-phi.vercel.app",
        "https://fes-website-chatbot-proto.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------
# 🧠 Embeddings cache
# ---------------------------------
EMBED_CACHE = {}
# ---------------------------------
# 🎓 University Tiers (FINAL – SOURCE OF TRUTH)
# ---------------------------------

TIERS = {
    1: [
        ("Edinburgh Napier University", "£17,500 – £21,500"),
        ("University of Portsmouth", "£16,200 – £19,200"),
        ("University of South Wales", "£13,700 – £18,000"),
        ("University of East Anglia", "£13,700 – £18,000"),
        ("University of Dundee", "£13,700 – £18,000"),
        ("Robert Gordon University", "£13,700 – £18,000"),
        ("Manchester Metropolitan University (MMU)", "£13,700 – £18,000"),
        ("Liverpool John Moores University", "£13,700 – £18,000"),
        ("University of Stirling", "£13,700 – £18,000"),
        ("Anglia Ruskin University", "£13,700 – £18,000"),
        ("University of Essex", "£13,700 – £18,000"),
        ("University of Northampton", "£13,700 – £18,000"),
        ("Nottingham Trent University", "£13,700 – £18,000"),
    ],

    2: [
        ("Sheffield Hallam University (SHU)", "£16,385 / £18,820"),
        ("University of Westminster", "£17,000"),
        ("Ulster University", "£13,800 (Birmingham & Manchester), £15,450 (London)"),
        ("SRUC – Scotland’s Rural College", "£19,000"),
        ("Prifysgol Aberystwyth University", "£19,700 / £21,000"),
        ("Regent College London", "£15,950"),
        ("Middlesex University London", "£19,200"),
        ("Heriot-Watt University", "£19,456"),
        ("University of Hull (Main Campus)", "£16,000"),
        ("De Montfort University (DMU)", "£17,950"),
        ("University of Bradford", "£20,000 – £22,000"),
    ],

    3: [
        ("Kingston University London", "£16,600"),
        ("Birmingham City University (BCU)", "£18,600"),
        ("Swansea University", "£22,750"),
        ("Wrexham Glyndwr University", "£12,500"),
        ("Bath Spa University", "£15,905"),
        ("Cardiff Metropolitan University", "£17,600"),
    ],
}
def detect_tier_query(query: str):
    q = query.lower()

    # -----------------------------
    # 1️⃣ Detect tier number
    # -----------------------------
    tier = None
    if "tier 1" in q or "tier1" in q:
        tier = 1
    elif "tier 2" in q or "tier2" in q:
        tier = 2
    elif "tier 3" in q or "tier3" in q:
        tier = 3

    if not tier:
        return None

    # -----------------------------
    # 2️⃣ Must be a LISTING request
    # -----------------------------
    list_keywords = [
        "universities", "unis", "list", "show", "give",
        "which universities", "universities in", "fes universities"
    ]

    if not any(k in q for k in list_keywords):
        return None

    # -----------------------------
    # 3️⃣ Negation handling
    # -----------------------------
    negations = [
        "not tier", "except tier", "other than tier",
        "exclude tier", "without tier"
    ]

    if any(n in q for n in negations):
        return None

    return tier

def embed_query(query: str):
    if query in EMBED_CACHE:
        return EMBED_CACHE[query]
    emb = client.embeddings.create(model="text-embedding-3-small", input=query)
    vec = emb.data[0].embedding
    EMBED_CACHE[query] = vec
    return vec

# ---------------------------------
# 🔎 Pinecone search
# ---------------------------------
def pinecone_search(query_vector, top_k: int = 3):
    res = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )
    matches = []
    for m in getattr(res, "matches", []) or []:
        matches.append({"id": m.id, "score": m.score, "metadata": m.metadata or {}})
    return matches

# ---------------------------------
# 🧩 Build context (same as your standalone)
# ---------------------------------
def build_context(matches, max_intro_chars: int = 300) -> str:
    lines = []
    for m in matches:
        meta = m["metadata"] or {}
        if "branch" in meta:
            # Contact entry
            branch = meta.get("branch", "Unknown Branch")
            intro = (meta.get("intro", "") or "")[:max_intro_chars]
            if len(meta.get("intro", "") or "") > max_intro_chars:
                intro += "..."
            address = meta.get("address", "") or ""
            phone = meta.get("phone", "")
            if isinstance(phone, list):
                phone = ", ".join([p for p in phone if p])
            email = meta.get("email", "") or ""
            link = meta.get("link", "No link") or "No link"
            lines.append(f"[Contact: {branch}] (Score: {m['score']:.4f})")
            lines.append(
                f"{intro}\nAddress: {address}\nPhone: {phone}\nEmail: {email}\nLink: {link}"
            )
        else:
            # Blog entry
            title = meta.get("title") or meta.get("slug") or "Untitled"
            chunk = meta.get("chunk") or meta.get("content") or ""
            snippet = re.sub(r"\s+", " ", chunk).strip()[:1000]
            lines.append(f"[Blog: {title}] (Score: {m['score']:.4f})")
            lines.append(snippet)
        lines.append("-" * 40)
    return "\n".join(lines)

# ---------------------------------
# 🗣️ System instructions
# ---------------------------------
SYSTEM_INSTRUCTIONS = """
You are Mentora, the friendly and professional FES virtual counsellor.  

## Role & Identity
- Always act as part of **FES** (never mention other organizations).  
- Be warm, supportive, and professional like a real study-abroad counsellor.  

## Data Sources (via Pinecone)
- **University Lists** → all universities FES works with, grouped by country.  
- **University Details** → basic info like programs, ranking, and location.  
- **Blogs** → general study-abroad guides, tips, and articles.  
- **Contacts** → FES branches, phone numbers, and emails.  

## Query Rules
- If the user asks about universities in a country (e.g., “unis in Ireland” or “universities FES deals with in UK”) → **always return the full list of universities FES has for that country** (not just some).  
- If the user asks about a specific university → show its details in a structured format.  
- If the user asks for general study-abroad guidance → use blog content.  
- If the user asks for FES contact info or branches → show contacts.  
- If information is missing → reply: *“I don’t have that right now, but I can guide you further if you share more details.”*  

## Formatting & Tone
- Use **headings, bullets, and emojis**.  
- **Short & direct** → for contacts.  
- **Structured & supportive** → for guidance or university details.  

## Contact Info Standard
- Start with: *“We have FES branches in many cities such as Rawalpindi, Peshawar, Karachi, and more.”*  
- Always include: **info@fespak.com**  
- Highlight Lahore Head Office:  
  - Branch: Lahore Head Office  
  - Address: Office # 31/2, Upper Ground, Mall of Lahore, 172 Tufail Road, Cantt Lahore  
  - Phone: +92 345 8454787  
  - Email: info@fespak.com  
  - Link: https://fespak.com/our-branches/lahore-head-office/  
- End with: *“For specific branch information, you can ask about a particular branch, for example, ‘FES Rawalpindi contact’.”*  

## University Info Standard
- Answer in max 5–6 bullets with 3 clear sections:  
  🎓 **Well-Known Programs**  
  🌟 **Highlights**  
  🤝 **How FES Can Help** → Offer letters, university scholarships, visa support, pre-departure counselling  
- End with: *“Want to study here? FES can guide you through every step.”*  
"""


# ---------------------------------
# 🔑 Contact detection + selection
# ---------------------------------
CONTACT_KEYWORDS = ["contact", "phone", "email", "branch", "address", "office", "call", "number"]
CITY_HINTS = ["lahore", "karachi", "islamabad", "rawalpindi", "multan", "peshawar", "quetta", "faisalabad"]

def is_contact_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in CONTACT_KEYWORDS)

def format_contact(meta: dict) -> str:
    branch = meta.get("branch", "Unknown Branch")
    intro = meta.get("intro", "")
    address = meta.get("address", "")
    phone = meta.get("phone", "")
    if isinstance(phone, list):
        phone = ", ".join([p for p in phone if p])
    email = meta.get("email", "")
    link = meta.get("link", "No link")
    return (
        f"{branch}\n"
        f"{intro}\n"
        f"Address: {address}\n"
        f"Phone: {phone}\n"
        f"Email: {email}\n"
        f"Link: {link}\n"
    )

def pick_best_contact_match(matches: list, query: str) -> Optional[dict]:
    """Prefer a contact match whose branch/address mentions the city in the query (e.g., 'Lahore')."""
    q = query.lower()
    contacts = [m for m in matches if "branch" in (m.get("metadata") or {})]

    if not contacts:
        return None

    # If query mentions a known city, try to match
    city_in_query = None
    for city in CITY_HINTS:
        if city in q:
            city_in_query = city
            break

    if city_in_query:
        for m in contacts:
            meta = m["metadata"] or {}
            branch = (meta.get("branch") or "").lower()
            address = (meta.get("address") or "").lower()
            intro = (meta.get("intro") or "").lower()
            if city_in_query in branch or city_in_query in address or city_in_query in intro:
                return m

    # Fallback: highest score contact
    return max(contacts, key=lambda x: x["score"]) if contacts else None

def fast_contact_response(matches: list, query: str) -> Optional[str]:
    m = pick_best_contact_match(matches, query)
    if not m:
        return None
    meta = m["metadata"] or {}
    return format_contact(meta)

# ---------------------------------
# 💬 Non-streaming answer (kept for /chat)
# ---------------------------------
def generate_answer(user_query: str, context_text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": f"User Query: {user_query}\n\nContext:\n{context_text}"},
        ],
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------
# 🌊 Streaming generator
# ---------------------------------
def generate_answer_stream(user_query: str, context_text: str) -> Iterable[str]:
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": f"User Query: {user_query}\n\nContext:\n{context_text}"},
        ],
    )
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content
        except Exception:
            # Safeguard against occasional malformed chunks
            continue

# ---------------------------------
# 🚀 RAG orchestration (shared)
# ---------------------------------
def run_rag(query: str, top_k: int = 3):
    qvec = embed_query(query)
    matches = pinecone_search(qvec, top_k=top_k)
    return matches

# ---------------------------------
# 🛣️ Endpoints
# ---------------------------------
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(req: ChatRequest):
    """Non-streaming JSON endpoint (useful for debugging)."""
    try:
        start = time.time()

        # -------------------------------
        # 🎯 TIER SHORT-CIRCUIT (FIRST)
        # -------------------------------
        tier = detect_tier_query(req.query)
        if tier:
            lines = []
            for uni, fee in TIERS[tier]:
                if fee:
                    lines.append(f"• **{uni}** — {fee}")
                else:
                    lines.append(f"• **{uni}**")

            return {
                "response": (
                    f"🎓 **Tier {tier} Universities (FES Partner Universities – UK)**\n\n"
                    + "\n".join(lines)
                    + "\n\n👉 *FES can assist with offers, scholarships & visa processing.*"
                )
            }

        # -------------------------------
        # 🔍 NORMAL RAG FLOW (UNCHANGED)
        # -------------------------------
        matches = run_rag(req.query, top_k=3)
        if not matches:
            return {"response": "No relevant info found."}

        if is_contact_query(req.query):
            fast = fast_contact_response(matches, req.query)
            if fast:
                elapsed = time.time() - start
                return {
                    "response": f"{fast}[Retrieved {len(matches)} chunks | Response time: {elapsed:.2f}s]"
                }

        context = build_context(matches)
        answer = generate_answer(req.query, context)
        elapsed = time.time() - start

        return {
            "response": f"{answer}\n\n[Retrieved {len(matches)} chunks | Response time: {elapsed:.2f}s]"
        }

    except Exception as e:
        logging.exception("Error in /chat")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process query: {str(e)}"}
        )

@app.get("/stream")
async def stream(q: str):
    """
    Streaming SSE endpoint (RAG + GPT tokens).
    """
    async def event_generator():
        start = time.time()
        try:
            # -------------------------------
            # 🎯 TIER SHORT-CIRCUIT (FIRST)
            # -------------------------------
            tier = detect_tier_query(q)
            if tier:
                yield {
                    "event": "message",
                    "data": f"🎓 **Tier {tier} Universities (FES Partner Universities – UK)**\n\n"
                }

                for uni, fee in TIERS[tier]:
                    line = f"• {uni}"
                    if fee:
                        line += f" — {fee}"
                    yield {"event": "message", "data": line + "\n"}
                    await asyncio.sleep(0)

                yield {
                    "event": "message",
                    "data": "\n👉 *FES can guide you end-to-end.*"
                }
                yield {"event": "message", "data": "[DONE]"}
                return

            # -------------------------------
            # 🔍 NORMAL RAG FLOW (UNCHANGED)
            # -------------------------------
            matches = run_rag(q, top_k=3)

            if not matches:
                yield {"event": "message", "data": "No relevant info found."}
                yield {"event": "message", "data": "[DONE]"}
                return

            if is_contact_query(q):
                fast = fast_contact_response(matches, q)
                if fast:
                    yield {"event": "message", "data": fast}
                    yield {
                        "event": "message",
                        "data": f"[Retrieved {len(matches)} chunks | Response time: {time.time()-start:.2f}s]"
                    }
                    yield {"event": "message", "data": "[DONE]"}
                    return

            context = build_context(matches)

            for token in generate_answer_stream(q, context):
                yield {"event": "message", "data": token}
                await asyncio.sleep(0)

            yield {
                "event": "message",
                "data": f"\n\n[Retrieved {len(matches)} chunks | Response time: {time.time()-start:.2f}s]"
            }
            yield {"event": "message", "data": "[DONE]"}

        except Exception as e:
            logging.exception("Error in /stream")
            yield {
                "event": "message",
                "data": f"⚠️ Something went wrong while answering.\n\nDetails: {str(e)}"
            }
            yield {"event": "message", "data": "[DONE]"}

    return EventSourceResponse(event_generator())
