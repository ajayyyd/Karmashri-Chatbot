-- scripts/init_schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kerala_go_documents (
    go_number TEXT PRIMARY KEY,
    go_type TEXT,
    order_date DATE,
    department TEXT,
    dept_code TEXT,
    source_file TEXT,
    total_pages INT,
    extraction_mode TEXT,
    cited_orders JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kerala_go_nodes (
    id TEXT PRIMARY KEY,
    go_number TEXT REFERENCES kerala_go_documents(go_number) ON DELETE CASCADE,
    section TEXT NOT NULL,
    clause_marker TEXT,
    parent_path TEXT,
    is_clause BOOLEAN,
    text TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_kerala_go_nodes_tsv 
ON kerala_go_nodes USING gin (to_tsvector('english', text));

CREATE INDEX IF NOT EXISTS idx_go_docs_dept_code ON kerala_go_documents (dept_code);
CREATE INDEX IF NOT EXISTS idx_go_docs_order_date ON kerala_go_documents (order_date);