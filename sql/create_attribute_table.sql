-- chem
CREATE TABLE attribute_vectors (
    -- The Socrata dataset four-by-four.
    dataset_id TEXT NOT NULL PRIMARY KEY,
    attribute_name TEXT NOT NULL,
    -- Embedding vector.
    emb BLOB NOT NULL
);
