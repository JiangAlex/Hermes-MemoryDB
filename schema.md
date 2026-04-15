# Memory Database Schema

## SQLite Vector Database for OpenClaw

### Tables

#### memories
Long-term curated memories (corresponds to MEMORY.md)
- `id` INTEGER PRIMARY KEY
- `category` TEXT - e.g., "project", "person", "preference", "skill"
- `title` TEXT - brief title
- `content` TEXT - the actual memory content
- `tags` TEXT - comma-separated tags for filtering
- `created_at` TIMESTAMP
- `updated_at` TIMESTAMP

#### daily_notes
Raw daily logs (corresponds to memory/YYYY-MM-DD.md)
- `id` INTEGER PRIMARY KEY
- `date` TEXT UNIQUE - YYYY-MM-DD format
- `content` TEXT - raw markdown content
- `created_at` TIMESTAMP

#### memory_embeddings
Vector embeddings for semantic search (using sqlite-vector)
- `id` INTEGER PRIMARY KEY
- `memory_id` INTEGER - FK to memories.id
- `memory_type` TEXT - 'memory' or 'daily_note'
- `embedding` BLOB - binary vector data (1536 dim for text-embedding-3-small)
- `created_at` TIMESTAMP

### Indexes
- `idx_memories_category` on memories(category)
- `idx_memories_tags` on memories(tags)
- `idx_daily_notes_date` on daily_notes(date)
- `idx_embeddings_memory_id` on memory_embeddings(memory_id)

### Vector Search
Using sqlite-vector extension for similarity search.
Fallback to LIKE search if vector extension unavailable.