import os
import json
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

NOTES_DIR = Path("./data/notes")
TASKS_DIR = Path("./data/tasks")
TASKS_FILE = TASKS_DIR / "tasks.json"


def ensure_dirs() -> None:
    """Create notes and tasks directories if they do not exist."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    if not TASKS_FILE.exists():
        TASKS_FILE.write_text(json.dumps([], indent=2))


def save_uploaded_files(uploaded_files) -> List[str]:
    """Persist uploaded files into the notes directory."""
    ensure_dirs()
    saved = []
    for f in uploaded_files:
        dest = NOTES_DIR / f.name
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        saved.append(str(dest))
    return saved


def load_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            parts.append(txt)
    return "\n\n".join(parts)


def load_text_from_file(path: Path) -> str:
    """Load text from a file, handling PDFs and plain text."""
    if path.suffix.lower() == ".pdf":
        return load_text_from_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_all_notes() -> Dict[str, str]:
    """Load all supported notes from the notes directory."""
    ensure_dirs()
    docs = {}
    for p in NOTES_DIR.iterdir():
        if p.is_file() and (p.suffix.lower() == ".pdf" or p.suffix.lower() in {".txt", ".md", ".csv"}):
            try:
                docs[p.name] = load_text_from_file(p)
            except Exception as e:
                docs[p.name] = f"[ERROR loading {p.name}: {e}]"
    return docs


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for context retrieval."""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def build_context(notes: Dict[str, str], max_chars: int = 8000) -> str:
    """Build a single context string from loaded notes, respecting a max length."""
    pieces = []
    total = 0
    for name, text in notes.items():
        header = f"\n--- SOURCE: {name} ---\n"
        if total + len(header) + len(text) > max_chars:
            break
        pieces.append(header + text)
        total += len(header) + len(text)
    return "\n".join(pieces)


def load_tasks() -> List[Dict]:
    """Load tasks from the local JSON store."""
    ensure_dirs()
    try:
        return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_tasks(tasks: List[Dict]) -> None:
    """Persist tasks to the local JSON store."""
    ensure_dirs()
    TASKS_FILE.write_text(json.dumps(tasks, indent=2), encoding="utf-8")


def get_api_client():
    """Return the configured LLM client and provider name."""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if openai_key:
        from openai import OpenAI
        return OpenAI(api_key=openai_key), "openai"
    if anthropic_key:
        import anthropic
        return anthropic.Anthropic(api_key=anthropic_key), "anthropic"
    raise RuntimeError("Missing OPENAI_API_KEY or ANTHROPIC_API_KEY in environment.")


GROUNDED_SYSTEM_PROMPT = (
    "You are a strict study and task coach. "
    "Answer ONLY using the provided uploaded notes and documents. "
    "If the answer is not present in the provided context, explicitly state: "
    "'I cannot verify this from your notes.' Do not hallucinate or use outside knowledge. "
    "When asked to plan or break down work, use only the context available. "
    "Keep responses concise and structured."
)


def query_llm(user_message: str, context: str, provider: str = "openai", model: str | None = None) -> str:
    """Send a grounded query to the configured LLM provider."""
    client, detected = get_api_client()
    if provider == "openai":
        model = model or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_message}"},
            ],
        )
        return response.choices[0].message.content or ""
    else:
        model = model or "claude-3-5-haiku-20241022"
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.2,
            system=GROUNDED_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_message}"}
            ],
        )
        return response.content[0].text if response.content else ""
