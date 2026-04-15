#!/usr/bin/env python3
"""
Memory Database with Hybrid Search (Keyword + Embedding)
Supports local (sentence-transformers) and API providers (MiniMax, OpenAI, Ollama)
"""

import sqlite3
import os
import re
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path("/home/node/.openclaw/workspace/memory")
DB_PATH = Path("/home/node/.openclaw/workspace/memory/memory.db")

# Default embedding config (can be overridden via env/API)
DEFAULT_EMBEDDING_MODEL = "embo-01"
DEFAULT_EMBEDDING_DIM = 1536
DEFAULT_EMBEDDING_API = "https://api.minimaxi.com/v1/embeddings"

# Local embedding model (sentence-transformers)
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LOCAL_DIM = 384

def get_db():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(embedding_dim=DEFAULT_EMBEDDING_DIM):
    """Initialize database schema with vector support."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Memories table (long-term)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            tags TEXT,
            embedding_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Daily notes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            embedding_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Vector embeddings table (using JSON for storage since sqlite-vec may not be available)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER,
            memory_type TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedding_model TEXT,
            embedding_dim INTEGER,
            vector_data BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    """)
    
    # Config table for embeddings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            provider TEXT NOT NULL,
            api_url TEXT NOT NULL,
            api_key TEXT,
            model TEXT NOT NULL,
            dimensions INTEGER DEFAULT 1536,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_notes_date ON daily_notes(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_memory ON memory_embeddings(memory_id)")
    
    conn.commit()
    print(f"Database initialized at {DB_PATH}")
    return conn

def get_embedding_api_key():
    """Get API key from environment or config."""
    # Try various env vars
    for var in ["MINIMAX_API_KEY", "OPENAI_API_KEY", "API_KEY"]:
        key = os.environ.get(var)
        if key:
            return key
    # Try to get from openclaw config (read-only, no key exposed)
    return os.environ.get("MEMORY_EMBEDDING_KEY", "")

def get_embedding_config():
    """Get embedding configuration from database."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM embedding_config WHERE id = 1")
    row = cursor.fetchone()
    if row:
        return dict(row)
    
    # Default config
    return {
        "provider": "minimax",
        "api_url": os.environ.get("EMBEDDING_API_URL", DEFAULT_EMBEDDING_API),
        "api_key": get_embedding_api_key(),
        "model": os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "dimensions": DEFAULT_EMBEDDING_DIM,
        "local_model": os.environ.get("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_MODEL)
    }

def set_embedding_config(provider, api_url, api_key, model, dimensions, local_model=None):
    """Update embedding configuration."""
    conn = get_db()
    cursor = conn.cursor()
    local_model = local_model or DEFAULT_LOCAL_MODEL
    cursor.execute("""
        INSERT OR REPLACE INTO embedding_config (id, provider, api_url, api_key, model, dimensions, local_model, updated_at)
        VALUES (1, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (provider, api_url, api_key, model, dimensions, local_model))
    conn.commit()
    print(f"Embedding config updated: {provider} ({model}, {dimensions}dim)")

def generate_embedding(text, api_url=None, api_key=None, model=None):
    """Generate embedding for text using OpenAI-compatible API (MiniMax, Ollama, etc.)."""
    config = get_embedding_config()
    api_url = api_url or config["api_url"]
    api_key = api_key or config["api_key"]
    model = model or config["model"]
    
    if not api_key:
        print("Warning: No API key configured for embeddings")
        return None
    
    import urllib.request
    import urllib.error
    
    payload = {
        "texts": [text[:8000]],  # Truncate long text, MiniMax uses "texts" array
        "model": model
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            # MiniMax returns {"vectors": [[...]], "base_resp": {...}}
            vectors = result.get("vectors", [])
            if vectors and len(vectors) > 0:
                return vectors[0]  # First embedding vector
            print("No vectors in response:", result)
            return None
    except Exception as e:
        print(f"Embedding API error: {e}")
        return None

def generate_local_embedding(text, model_name=None):
    """Generate embedding using local sentence-transformers model."""
    config = get_embedding_config()
    model_name = model_name or config.get("local_model", DEFAULT_LOCAL_MODEL)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Cache the model to avoid reloading
        if not hasattr(generate_local_embedding, "_model"):
            generate_local_embedding._model = SentenceTransformer(model_name)
        
        embedding = generate_local_embedding._model.encode(text[:8000])
        return embedding.tolist()
    except ImportError:
        print("Error: sentence-transformers not installed. Run: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"Local embedding error: {e}")
        return None

def generate_embedding_auto(text, provider=None):
    """Auto-select embedding provider based on config or override."""
    config = get_embedding_config()
    provider = provider or config.get("provider", "local")
    
    if provider == "local":
        return generate_local_embedding(text)
    else:
        return generate_embedding(text)

def compute_content_hash(content):
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def get_embedding_dim(embedding):
    """Get dimensionality of embedding."""
    return len(embedding) if embedding else 0

def store_embedding(memory_id, memory_type, content, embedding):
    """Store embedding for memory item."""
    conn = get_db()
    cursor = conn.cursor()
    
    content_hash = compute_content_hash(content)
    dimensions = get_embedding_dim(embedding)
    
    # Store as binary blob (can be changed to use sqlite-vec if available)
    vector_bytes = json.dumps(embedding).encode("utf-8")
    
    # Check if embedding already exists
    cursor.execute("""
        SELECT id FROM memory_embeddings 
        WHERE memory_id = ? AND memory_type = ? AND content_hash = ?
    """, (memory_id, memory_type, content_hash))
    
    existing = cursor.fetchone()
    if existing:
        print(f"Embedding already exists for {memory_type}:{memory_id}")
        return existing["id"]
    
    config = get_embedding_config()
    cursor.execute("""
        INSERT INTO memory_embeddings 
        (memory_id, memory_type, content_hash, embedding_model, embedding_dim, vector_data)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (memory_id, memory_type, content_hash, config["model"], dimensions, vector_bytes))
    
    conn.commit()
    print(f"Stored embedding for {memory_type}:{memory_id} ({dimensions}dim)")
    return cursor.lastrowid

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

def search_memories(query, limit=10, use_vector=True, use_keyword=True, hybrid_alpha=0.5):
    """Hybrid search: combine keyword (BM25/LIKE) + vector similarity.
    
    Args:
        query: Search query string
        limit: Max results to return
        use_vector: Enable vector search
        use_keyword: Enable keyword search
        hybrid_alpha: Weight for vector score (1-alpha for keyword). 0.5 = equal weight.
    
    Returns:
        List of dicts with memory data and combined scores
    """
    conn = get_db()
    cursor = conn.cursor()
    results_dict = {}  # id -> result, for merging
    
    # 1. Keyword search (text match)
    if use_keyword:
        cursor.execute("""
            SELECT id, title, content, category, tags, updated_at
            FROM memories
            WHERE content LIKE ? OR title LIKE ? OR tags LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%", limit * 2))
        
        keyword_results = cursor.fetchall()
        max_keyword_score = 1.0
        for row in keyword_results:
            # Simple binary scoring: 1 if match, 0.5 if partial
            score = 1.0 if (query.lower() in row["title"].lower() if row["title"] else False) else 0.5
            results_dict[row["id"]] = {
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "category": row["category"],
                "tags": row["tags"],
                "keyword_score": score,
                "vector_score": 0.0,
                "source": "keyword"
            }
    
    # 2. Vector search (semantic similarity)
    if use_vector:
        # Use auto provider selection (local or API)
        embedding = generate_embedding_auto(query)
        if embedding:
            config = get_embedding_config()
            cursor.execute("""
                SELECT me.*, m.title, m.content, m.category, m.tags
                FROM memory_embeddings me
                JOIN memories m ON me.memory_id = m.id
                WHERE me.memory_type = 'memory'
            """)
            
            vector_results = cursor.fetchall()
            for row in vector_results:
                stored_vector = json.loads(row["vector_data"].decode("utf-8"))
                similarity = cosine_similarity(embedding, stored_vector)
                
                if row["memory_id"] in results_dict:
                    results_dict[row["memory_id"]]["vector_score"] = similarity
                    results_dict[row["memory_id"]]["source"] = "hybrid"
                else:
                    results_dict[row["memory_id"]] = {
                        "id": row["memory_id"],
                        "title": row["title"],
                        "content": row["content"],
                        "category": row["category"],
                        "tags": row["tags"],
                        "keyword_score": 0.0,
                        "vector_score": similarity,
                        "source": "vector"
                    }
    
    # 3. Combine scores (hybrid ranking)
    final_results = []
    for rid, r in results_dict.items():
        # Normalize: keyword 0-1, vector 0-1, combined weighted
        kw = r["keyword_score"]
        vec = r["vector_score"]
        r["score"] = (hybrid_alpha * vec) + ((1 - hybrid_alpha) * kw) if (kw > 0 or vec > 0) else 0
        r["keyword_score"] = kw
        r["vector_score"] = vec
        final_results.append(r)
    
    # Sort by combined score
    final_results.sort(key=lambda x: x["score"], reverse=True)
    return final_results[:limit]

def search_daily_notes(query, limit=10, use_vector=True, use_keyword=True, hybrid_alpha=0.5):
    """Hybrid search for daily notes."""
    conn = get_db()
    cursor = conn.cursor()
    results_dict = {}
    
    if use_keyword:
        cursor.execute("""
            SELECT id, date, content
            FROM daily_notes
            WHERE content LIKE ?
            ORDER BY date DESC
            LIMIT ?
        """, (f"%{query}%", limit * 2))
        
        for row in cursor.fetchall():
            results_dict[row["id"]] = {
                "id": row["id"],
                "date": row["date"],
                "content": row["content"],
                "keyword_score": 1.0,
                "vector_score": 0.0,
                "source": "keyword"
            }
    
    if use_vector:
        embedding = generate_embedding_auto(query)
        if embedding:
            cursor.execute("""
                SELECT me.*, d.date, d.content
                FROM memory_embeddings me
                JOIN daily_notes d ON me.memory_id = d.id
                WHERE me.memory_type = 'daily'
            """)
            
            for row in cursor.fetchall():
                stored_vector = json.loads(row["vector_data"].decode("utf-8"))
                similarity = cosine_similarity(embedding, stored_vector)
                
                if row["memory_id"] in results_dict:
                    results_dict[row["memory_id"]]["vector_score"] = similarity
                    results_dict[row["memory_id"]]["source"] = "hybrid"
                else:
                    results_dict[row["memory_id"]] = {
                        "id": row["memory_id"],
                        "date": row["date"],
                        "content": row["content"],
                        "keyword_score": 0.0,
                        "vector_score": similarity,
                        "source": "vector"
                    }
    
    final_results = []
    for rid, r in results_dict.items():
        kw = r["keyword_score"]
        vec = r["vector_score"]
        r["score"] = (hybrid_alpha * vec) + ((1 - hybrid_alpha) * kw)
        r["keyword_score"] = kw
        r["vector_score"] = vec
        final_results.append(r)
    
    final_results.sort(key=lambda x: x["score"], reverse=True)
    return final_results[:limit]

def search_all(query, limit=10, hybrid_alpha=0.5):
    """Search both memories and daily notes."""
    mem_results = search_memories(query, limit, use_vector=True, use_keyword=True, hybrid_alpha=hybrid_alpha)
    note_results = search_daily_notes(query, limit, use_vector=True, use_keyword=True, hybrid_alpha=hybrid_alpha)
    
    combined = mem_results + note_results
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:limit]

def import_daily_notes():
    """Import daily notes from .md files."""
    conn = get_db()
    cursor = conn.cursor()
    
    count = 0
    for md_file in MEMORY_DIR.glob("*.md"):
        if md_file.name in ["schema.md", "memory_db.py"]:
            continue
        
        date_match = re.match(r'(\d{4}-\d{2}-\d{2})\.md', md_file.name)
        if not date_match:
            continue
        
        date_str = date_match.group(1)
        
        cursor.execute("SELECT id FROM daily_notes WHERE date = ?", (date_str,))
        if cursor.fetchone():
            print(f"Skipping {md_file.name} (already imported)")
            continue
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cursor.execute(
            "INSERT INTO daily_notes (date, content) VALUES (?, ?)",
            (date_str, content)
        )
        count += 1
        print(f"Imported {md_file.name}")
    
    conn.commit()
    print(f"Imported {count} daily notes")
    return count

def import_memory_md():
    """Import MEMORY.md as long-term memories."""
    memory_md = Path("/home/node/.openclaw/workspace/MEMORY.md")
    if not memory_md.exists():
        print("MEMORY.md not found, skipping")
        return 0
    
    conn = get_db()
    cursor = conn.cursor()
    
    content = memory_md.read_text(encoding='utf-8')
    sections = parse_md_file(memory_md)
    
    count = 0
    current_category = "general"
    
    for section in sections["sections"]:
        title = section["title"]
        section_content = '\n'.join(section["content"]).strip()
        
        if not section_content:
            continue
        
        if title.startswith("### "):
            category = "project"
            title = title[4:].strip()
        elif title.startswith("## "):
            category = "general"
            title = title[3:].strip()
            current_category = title.lower()
        else:
            category = current_category
        
        tags = extract_tags(section_content)
        
        # Check if already exists
        cursor.execute("SELECT id FROM memories WHERE title = ?", (title,))
        existing = cursor.fetchone()
        if existing:
            print(f"Skipping {title} (already exists)")
            continue
        
        cursor.execute("""
            INSERT INTO memories (category, title, content, tags)
            VALUES (?, ?, ?, ?)
        """, (category, title, section_content, tags))
        count += 1
        print(f"Imported memory: {title}")
    
    conn.commit()
    print(f"Imported {count} memories from MEMORY.md")
    return count

def parse_md_file(file_path):
    """Parse markdown file and extract sections."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = []
    current_section = {"title": "", "content": []}
    
    for line in content.split('\n'):
        if line.startswith('## '):
            if current_section["title"] or current_section["content"]:
                sections.append(current_section)
            current_section = {"title": line[3:].strip(), "content": []}
        else:
            current_section["content"].append(line)
    
    if current_section["title"] or current_section["content"]:
        sections.append(current_section)
    
    return {
        "title": file_path.stem,
        "sections": sections,
        "full_content": content
    }

def extract_tags(content):
    """Extract tags from content."""
    tags = []
    for line in content.split('\n'):
        match = re.search(r'#(\w+)', line)
        if match:
            tags.append(match.group(1))
    return ','.join(tags) if tags else None

def embed_all_memories(use_local=True):
    """Generate and store embeddings for all memories.
    
    Args:
        use_local: If True, use local sentence-transformers; otherwise use API
    """
    conn = get_db()
    cursor = conn.cursor()
    
    # Get all memories without embeddings
    cursor.execute("""
        SELECT m.id, m.title, m.content, m.category, m.tags
        FROM memories m
        LEFT JOIN memory_embeddings me ON m.id = me.memory_id AND me.memory_type = 'memory'
        WHERE me.id IS NULL
    """)
    
    memories = cursor.fetchall()
    print(f"Found {len(memories)} memories to embed")
    
    provider = "local" if use_local else "api"
    print(f"Using provider: {provider}")
    
    for mem in memories:
        content = f"{mem['title']}\n{mem['content']}"
        embedding = generate_embedding_auto(content, provider=provider)
        if embedding:
            store_embedding(mem["id"], "memory", content, embedding)
        else:
            print(f"Failed to generate embedding for: {mem['title']}")
    
    conn.commit()
    return len(memories)

def embed_all_daily_notes(use_local=True):
    """Generate and store embeddings for all daily notes."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT d.id, d.date, d.content
        FROM daily_notes d
        LEFT JOIN memory_embeddings me ON d.id = me.memory_id AND me.memory_type = 'daily'
        WHERE me.id IS NULL
    """)
    
    notes = cursor.fetchall()
    print(f"Found {len(notes)} daily notes to embed")
    
    provider = "local" if use_local else "api"
    
    for note in notes:
        content = f"{note['date']}\n{note['content']}"
        embedding = generate_embedding_auto(content, provider=provider)
        if embedding:
            store_embedding(note["id"], "daily", content, embedding)
        else:
            print(f"Failed to generate embedding for: {note['date']}")
    
    conn.commit()
    return len(notes)

def embed_all(use_local=True):
    """Generate embeddings for both memories and daily notes."""
    mem_count = embed_all_memories(use_local=use_local)
    note_count = embed_all_daily_notes(use_local=use_local)
    print(f"Embedded {mem_count} memories and {note_count} daily notes")

def main():
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "init":
            init_db()
            return
        
        if cmd == "import":
            init_db()
            import_daily_notes()
            import_memory_md()
            return
        
        if cmd == "embed":
            use_local = "--local" in sys.argv
            init_db()
            embed_all(use_local=use_local)
            return
        
        if cmd == "search":
            query = sys.argv[2] if len(sys.argv) > 2 else ""
            hybrid_alpha = 0.5
            # Parse args: search [--hybrid alpha] [--vector-only] [--keyword-only] <query>
            use_vector = True
            use_keyword = True
            for i, arg in enumerate(sys.argv):
                if arg == "--vector-only":
                    use_keyword = False
                elif arg == "--keyword-only":
                    use_vector = False
                elif arg == "--hybrid" and i + 1 < len(sys.argv):
                    hybrid_alpha = float(sys.argv[i + 1])
            results = search_all(query, limit=10, hybrid_alpha=hybrid_alpha)
            for r in results:
                src = r.get("source", "unknown")
                print(f"[{r.get('category', 'unknown')}] {r.get('title', r.get('date', 'No title'))}")
                print(f"  Source: {src}, Score: {r.get('score', 0):.3f} (kw:{r.get('keyword_score', 0):.2f} vec:{r.get('vector_score', 0):.2f})")
                content_preview = r.get('content', '')[:150]
                print(f"  {content_preview}...")
                print()
            return
        
        if cmd == "config":
            if len(sys.argv) < 6:
                print("Usage: memory_db.py config <provider> <api_url> <model> <dimensions> [--local-model <model>]")
                print("  provider: local, minimax, openai, ollama")
                print("  example: memory_db.py config minimax https://api.minimaxi.com/v1/embeddings embo-01 1536")
                print("  example: memory_db.py config local http://localhost:11434/api/embeddings nomic-embed-text 768")
                return
            provider = sys.argv[2]
            api_url = sys.argv[3]
            model = sys.argv[4]
            dims = int(sys.argv[5])
            local_model = DEFAULT_LOCAL_MODEL
            # Check for local model override
            for i, arg in enumerate(sys.argv):
                if arg == "--local-model" and i + 1 < len(sys.argv):
                    local_model = sys.argv[i + 1]
            api_key = get_embedding_api_key()
            set_embedding_config(provider, api_url, api_key, model, dims, local_model)
            return
        
        if cmd == "status":
            config = get_embedding_config()
            print(f"Provider: {config.get('provider', 'minimax')}")
            print(f"API URL: {config.get('api_url', 'N/A')}")
            print(f"Model: {config.get('model', 'N/A')}")
            print(f"Dimensions: {config.get('dimensions', 'N/A')}")
            print(f"Local Model: {config.get('local_model', DEFAULT_LOCAL_MODEL)}")
            print(f"API Key set: {'Yes' if config.get('api_key') else 'No'}")
            
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM memories")
            print(f"Memories: {cursor.fetchone()['count']}")
            cursor.execute("SELECT COUNT(*) as count FROM daily_notes")
            print(f"Daily notes: {cursor.fetchone()['count']}")
            cursor.execute("SELECT COUNT(*) as count FROM memory_embeddings")
            print(f"Embeddings: {cursor.fetchone()['count']}")
            return
        
        if cmd == "test-embedding":
            # Test embedding generation
            test_text = "Hello world, this is a test"
            print(f"Testing with: '{test_text}'")
            
            # Test local
            local_emb = generate_local_embedding(test_text)
            if local_emb:
                print(f"Local embedding: {len(local_emb)}dim, first 5: {local_emb[:5]}")
            else:
                print("Local embedding failed")
            
            # Test API
            api_emb = generate_embedding(test_text)
            if api_emb:
                print(f"API embedding: {len(api_emb)}dim, first 5: {api_emb[:5]}")
            else:
                print("API embedding failed (no key or API error)")
            return
    
    # Default: init and import
    init_db()
    import_daily_notes()
    import_memory_md()
    print("\nDone! Run with:")
    print("  search [--hybrid 0.5] [--vector-only] [--keyword-only] <query>  - Hybrid search")
    print("  embed [--local]      - Generate embeddings (use --local for sentence-transformers)")
    print("  config <provider> <api_url> <model> <dims> - Configure embedding provider")
    print("  test-embedding       - Test embedding generation")
    print("  status               - Show config and stats")

if __name__ == "__main__":
    main()