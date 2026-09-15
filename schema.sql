-- Compass — application-owned schema.
--
-- This file covers ONLY conversation_history. LangGraph's four checkpoint
-- tables (checkpoints, checkpoint_writes, checkpoint_blobs,
-- checkpoint_migrations) are created by AsyncPostgresSaver.setup() and must
-- never be hand-written here — their schema is owned by the library and
-- changes between versions.
--
-- Apply with:  make migrate     (see Makefile)

CREATE TABLE IF NOT EXISTS conversation_history (
  thread_id       text PRIMARY KEY,

  -- Resolved from the caller's bearer token via getPersonalInfo (SPEC §2.3).
  -- Never accepted from a request body.
  user_id         int  NOT NULL,

  title           text,
  messages        jsonb NOT NULL DEFAULT '[]',
  awaiting_input  jsonb,
  status          text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

-- Serves GET /conversations (sidebar: this user's threads, newest first).
CREATE INDEX IF NOT EXISTS idx_conv_history_user_updated
  ON conversation_history (user_id, updated_at DESC);
