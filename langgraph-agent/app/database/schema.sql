-- ─────────────────────────────────────────────────────────────────────────────
-- schema.sql  —  Conversation History & Memory Database Schema
--
-- WHY THIS SCHEMA?
--   Every production LLM application needs to remember what users said.
--   This schema supports millions of users each with thousands of conversations.
--
-- TABLE HIERARCHY:
--   users
--     └── conversations  (one user → many conversations)
--           └── sessions    (one conversation → many sessions)
--                 └── messages  (one session → many messages)
--   users
--     └── memory       (long-term semantic memory per user)
--
-- HOW IT MAPS TO REAL USAGE:
--   user_123 opens your app
--   → creates conversation_001 ("RAG discussion")
--   → starts session_A (this browser tab / today's visit)
--   → sends 10 messages (stored in messages table)
--   → closes tab
--   Next day:
--   → creates session_B (new visit, same conversation)
--   → messages reference both session_B and conversation_001
--   → long-term memory about user's preferences stored in memory table
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable required extensions
-- pgvector: allows storing 1536-dimensional embeddings for semantic search
-- uuid-ossp: generates UUID primary keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: users
--
-- WHY THIS TABLE?
--   Central identity store. Every other table references user_id.
--   Allows per-user analytics, rate limiting, and personalisation.
--
-- FIELDS:
--   id          — UUID primary key (never expose to users, use display_name)
--   external_id — The ID from your auth provider (e.g. Keycloak, Auth0)
--   name        — Display name shown in UI
--   email       — For notifications (optional)
--   metadata    — Arbitrary JSON: timezone, plan tier, preferences
--   created_at  — When user first registered
--   last_seen_at — When user last made a request (for analytics)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id     TEXT        UNIQUE,                     -- from auth provider
    name            TEXT        NOT NULL DEFAULT 'Anonymous',
    email           TEXT        UNIQUE,
    metadata        JSONB       NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_external_id ON users(external_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: conversations
--
-- WHY THIS TABLE?
--   A user can have many separate conversations (topics).
--   "RAG discussion", "Fine-tuning help", "Career advice" are separate.
--   This lets you display a conversation list (like ChatGPT's sidebar).
--
-- FIELDS:
--   id             — UUID for this conversation
--   user_id        — FK to users table
--   title          — Auto-generated or user-named title (first user message)
--   summary        — LLM-generated summary of the whole conversation
--   message_count  — Cached count (avoids COUNT(*) queries)
--   total_tokens   — Running token budget used in this conversation
--   metadata       — Tags, model used, etc.
--   is_archived    — Soft delete / archive older conversations
--   created_at     — When conversation started
--   updated_at     — When last message was added
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           TEXT        NOT NULL DEFAULT 'New Conversation',
    summary         TEXT,                                   -- LLM-generated summary
    message_count   INT         NOT NULL DEFAULT 0,
    total_tokens    INT         NOT NULL DEFAULT 0,
    metadata        JSONB       NOT NULL DEFAULT '{}',
    is_archived     BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: sessions
--
-- WHY THIS TABLE?
--   A session is ONE visit / browser tab opening.
--   A conversation spans multiple sessions (you come back next day).
--   Session tracking helps with:
--     - Rate limiting per session
--     - Detecting abandoned sessions
--     - Analytics: average session length, messages per session
--
-- HOW session ≠ conversation:
--   conversation = the topic ("Tell me about RAG")
--   session      = the browser visit ("I talked about RAG on Monday 3pm-4pm")
--
-- FIELDS:
--   id               — UUID for this session
--   conversation_id  — FK to conversations
--   user_id          — Denormalized FK (avoids joins for auth checks)
--   thread_id        — LangGraph thread ID (for interrupt/resume)
--   started_at       — Session start time
--   ended_at         — NULL means session is still active
--   is_active        — TRUE while user is currently in this session
--   message_count    — Messages in this session
--   metadata         — Client info: browser, IP hash, device
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id               UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id  UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id          UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    thread_id        TEXT,                                  -- LangGraph thread_id
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at         TIMESTAMPTZ,
    is_active        BOOLEAN     NOT NULL DEFAULT TRUE,
    message_count    INT         NOT NULL DEFAULT 0,
    metadata         JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_conversation_id ON sessions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_thread_id ON sessions(thread_id);
CREATE INDEX IF NOT EXISTS idx_sessions_is_active ON sessions(is_active) WHERE is_active = TRUE;


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: messages
--
-- WHY THIS TABLE?
--   Every message (user and assistant) is stored permanently.
--   This is the core of conversation history.
--
-- CONTEXT WINDOW PROBLEM:
--   If a conversation has 10,000 messages, we obviously can't send all of
--   them to the LLM (context window = 128K tokens ≈ 100K words).
--   Solution: fetch recent N messages, or use semantic search via `embedding`
--   column to find the most RELEVANT past messages for the current query.
--
-- FIELDS:
--   id               — UUID for this message
--   conversation_id  — FK to conversations
--   session_id       — FK to sessions (which visit this message came from)
--   user_id          — Denormalized (fast filtering by user without joins)
--   role             — "user" | "assistant" | "system" | "tool"
--   content          — The actual message text
--   token_count      — Pre-computed tokens for context window management
--   embedding        — 1536-d vector for semantic similarity search
--                      (populated asynchronously after message is saved)
--   model_used       — Which LLM generated this (for fine-tuning analysis)
--   latency_ms       — How long the LLM took (for performance monitoring)
--   langfuse_trace_id — Link to Langfuse trace for this message
--   metadata         — RAG sources, tool calls, quality scores, etc.
--   created_at       — Message timestamp
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id                  UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id     UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    session_id          UUID        REFERENCES sessions(id) ON DELETE SET NULL,
    user_id             UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role                TEXT        NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content             TEXT        NOT NULL,
    token_count         INT         NOT NULL DEFAULT 0,
    embedding           VECTOR(1536),                       -- semantic search vector
    model_used          TEXT,                               -- e.g. "gpt-4o"
    latency_ms          INT,                                -- LLM response time
    langfuse_trace_id   TEXT,                               -- link to Langfuse
    metadata            JSONB       NOT NULL DEFAULT '{}',  -- sources, tools, scores
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Core access pattern: "give me the last N messages for conversation X"
CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
    ON messages(conversation_id, created_at DESC);

-- For semantic search: "find messages similar to this query"
-- ivfflat index with cosine distance for pgvector
CREATE INDEX IF NOT EXISTS idx_messages_embedding
    ON messages USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Fast user-level analytics
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: memory
--
-- WHY THIS TABLE?
--   Long-term memory is DIFFERENT from conversation history.
--
--   Conversation history = the raw transcript ("User: What is RAG? AI: RAG means...")
--   Long-term memory     = distilled facts ("User prefers Python examples",
--                          "User is a senior backend engineer",
--                          "User asked about chunking last Tuesday")
--
--   Long-term memory is retrieved semantically and injected into every
--   new request's system prompt, giving the LLM "personality memory".
--
-- MEMORY TYPES:
--   "preference"  — "User prefers concise answers"
--   "fact"        — "User's name is Manish, works at TechCorp"
--   "instruction" — "Always respond in bullet points"
--   "summary"     — "In session 3, user learned about vector databases"
--   "episodic"    — "Last week user asked about RAG pipeline design"
--
-- FIELDS:
--   id           — UUID
--   user_id      — FK to users (memory is per-user)
--   memory_type  — "preference" | "fact" | "instruction" | "summary" | "episodic"
--   content      — The actual memory text
--   embedding    — 1536-d vector for semantic retrieval
--   source_msg_id — Which message this memory was extracted from
--   importance   — 0.0-1.0 score (higher = injected first)
--   access_count — How many times this memory was retrieved (popularity)
--   expires_at   — NULL = permanent; set for temporary context
--   created_at   — When memory was formed
--   updated_at   — When memory was last modified
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    memory_type     TEXT        NOT NULL DEFAULT 'fact'
                                CHECK (memory_type IN ('preference', 'fact', 'instruction', 'summary', 'episodic')),
    content         TEXT        NOT NULL,
    embedding       VECTOR(1536),                           -- for semantic retrieval
    source_msg_id   UUID        REFERENCES messages(id) ON DELETE SET NULL,
    importance      FLOAT       NOT NULL DEFAULT 0.5
                                CHECK (importance >= 0.0 AND importance <= 1.0),
    access_count    INT         NOT NULL DEFAULT 0,
    expires_at      TIMESTAMPTZ,                            -- NULL = permanent
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Primary access: "get all memories for user X ordered by importance"
CREATE INDEX IF NOT EXISTS idx_memory_user_importance
    ON memory(user_id, importance DESC);

-- Semantic search over memories
CREATE INDEX IF NOT EXISTS idx_memory_embedding
    ON memory USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

-- Don't return expired memories
CREATE INDEX IF NOT EXISTS idx_memory_expires
    ON memory(expires_at) WHERE expires_at IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: conversation_summaries
--
-- WHY THIS TABLE?
--   When a conversation has 100+ messages, we can't load all of them.
--   Strategy: Summarize old message blocks and store the summary here.
--   When building context, load: [summary of old messages] + [last N messages]
--
-- FIELDS:
--   id               — UUID
--   conversation_id  — FK to conversations
--   summary_text     — LLM-generated summary of the message range
--   from_message_id  — First message included in this summary
--   to_message_id    — Last message included in this summary
--   message_range    — Human-readable "messages 1-50"
--   token_count      — How many tokens this summary uses
--   created_at       — When summary was generated
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id               UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id  UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    summary_text     TEXT        NOT NULL,
    from_message_id  UUID        REFERENCES messages(id) ON DELETE SET NULL,
    to_message_id    UUID        REFERENCES messages(id) ON DELETE SET NULL,
    message_range    TEXT,                                  -- e.g. "messages 1-50"
    token_count      INT         NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_summaries_conversation_id
    ON conversation_summaries(conversation_id, created_at DESC);


-- ─────────────────────────────────────────────────────────────────────────────
-- HELPER: update updated_at automatically on conversations table
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_memory_updated_at
    BEFORE UPDATE ON memory
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ─────────────────────────────────────────────────────────────────────────────
-- HELPER: increment conversation message_count and total_tokens
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION increment_conversation_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE conversations
    SET message_count = message_count + 1,
        total_tokens  = total_tokens + COALESCE(NEW.token_count, 0),
        updated_at    = NOW()
    WHERE id = NEW.conversation_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER messages_increment_conversation_stats
    AFTER INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION increment_conversation_stats();


-- ─────────────────────────────────────────────────────────────────────────────
-- SEED: default anonymous user (for testing without auth)
-- ─────────────────────────────────────────────────────────────────────────────
INSERT INTO users (id, external_id, name, email)
VALUES
    ('00000000-0000-0000-0000-000000000001', 'user_demo_1', 'Manish (Demo)', 'manish@demo.local'),
    ('00000000-0000-0000-0000-000000000002', 'user_demo_2', 'Priya (Demo)',  'priya@demo.local'),
    ('00000000-0000-0000-0000-000000000003', 'user_demo_3', 'Rahul (Demo)',  'rahul@demo.local')
ON CONFLICT (external_id) DO NOTHING;
