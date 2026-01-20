# DRAG API Documentation

## Overview

DRAG (Document Retrieval and Generation) is a query-focused document summarization system that combines semantic similarity, keyword matching, and document structure analysis to provide intelligent document search and summarization.

If you use DRAG in your research, please cite our paper:

**DRAG Me to the Source: Escaping the Chunking Trap in Extractive RAG**  

Osiris Terry, [arXiv preprint, 2026](https://arxiv.org/abs/xxxx.xxxxx)

```bibtex
@article{Terry2026,
  author    = {Osiris J. Terry, Christine Schubert Kabban, and Kenneth M. Hopkinson},
  title     = {DRAG Me to the Source: Escaping the Chunking Trap in Extractive RAG},
  year      = {2026},
  note      = {arXiv preprint},
  url       = {https://arxiv.org/abs/xxxx.xxxxx}
}
```

## Installation

```bash
pip install git+https://github.com/OJ-Pulp/DRAG.git
```

## Quick Start

```python
from DRAG import DRAG

# Initialize DRAG with a database
drag = DRAG(db_name="my_documents.db", db_path="/path/to/db")

# Upload a document
doc_id = drag.upload(text="Your document text here...", source="document.txt")

# Query documents
results = drag.query("What is the main topic?", max_sentences=5, max_documens=3)
```

## Class: DRAG

### Constructor

#### `__init__(db_name, db_path, alpha=0.55, beta=0.35, gamma=0.10)`

Initialize the DRAG system with a database and NLP toolkit.

**Parameters:**
- `db_name` (str): Name of the database file.
- `db_path` (str): Path to the directory containing the database.
- `alpha` (float, optional): Weight for semantic similarity in query scoring. Default: 0.55
- `beta` (float, optional): Weight for keyword matching in query scoring. Default: 0.35
- `gamma` (float, optional): Weight for document structure in query scoring. Default: 0.10

**Note:** The weights (alpha, beta, gamma) should sum to approximately 1.0 for balanced scoring.

**Example:**
```python
drag = DRAG(
    db_name="documents.db",
    db_path="/data/databases",
    alpha=0.6,  # Emphasize semantic similarity
    beta=0.3,   # Moderate keyword matching
    gamma=0.1   # Light document structure weighting
)
```

---

### Document Management Methods

#### `upload(text, source, metadata=None)`

Add a new document to the database with preprocessing and embedding generation.

**Parameters:**
- `text` (str): The full text content of the document.
- `source` (str): Identifier for the document source (e.g., filename, URL).
- `metadata` (dict, optional): Additional metadata to associate with the document. Will be merged with auto-generated metadata.

**Returns:**
- `str`: Unique document ID assigned to the uploaded document.

**Example:**
```python
doc_id = drag.upload(
    text="Long document text...",
    source="research_paper.pdf",
    metadata={"author": "John Doe", "year": 2024}
)
print(f"Document uploaded with ID: {doc_id}")
```

**Note:** The text is automatically processed to extract sentences, generate embeddings, compute similarity matrices, and extract keywords.

---

#### `delete(document_id)`

Remove a document and all its associated data from the database.

**Parameters:**
- `document_id` (str): Unique identifier of the document to delete.

**Returns:**
- None

**Example:**
```python
drag.delete("doc_12345")
```

**Note:** This removes the document, its sentences, embeddings, and all metadata.

---

#### `update_metadata(document_id, metadata)`

Update or replace metadata associated with a document.

**Parameters:**
- `document_id` (str): Unique identifier of the document.
- `metadata` (dict): New metadata dictionary to associate with the document.

**Returns:**
- None

**Example:**
```python
drag.update_metadata(
    "doc_12345",
    {"status": "reviewed", "tags": ["important", "research"]}
)
```

**Note:** This replaces the existing metadata entirely. To merge, retrieve current metadata first and combine before updating.

---

#### `batch(batch_size=100, offset=0)`

Retrieve documents from the database in paginated batches.

**Parameters:**
- `batch_size` (int, optional): Number of documents to retrieve per batch. Default: 100
- `offset` (int, optional): Starting position in the database (for pagination). Default: 0

**Returns:**
- `List[dict]`: List of document dictionaries containing document data and metadata.

**Example:**
```python
# Get first 100 documents
first_batch = drag.batch(batch_size=100, offset=0)

# Get next 100 documents
second_batch = drag.batch(batch_size=100, offset=100)

# Process all documents
offset = 0
while True:
    batch = drag.batch(batch_size=50, offset=offset)
    if not batch:
        break
    # Process batch...
    offset += 50
```

---

### Database Management Methods

#### `change_db(db_path, db_name)`

Switch to a different database.

**Parameters:**
- `db_path` (str): Path to the directory containing the new database.
- `db_name` (str): Name of the new database file.

**Returns:**
- None

**Example:**
```python
drag.change_db("/data/databases", "archive_documents.db")
```

**Note:** This closes the current database connection before opening the new one.

---

#### `delete_db(db_path, db_name)`

Permanently delete a database file.

**Parameters:**
- `db_path` (str): Path to the directory containing the database.
- `db_name` (str): Name of the database file to delete.

**Returns:**
- None

**Example:**
```python
drag.delete_db("/data/databases", "old_documents.db")
```

**⚠️ Warning:** This operation cannot be undone. All documents and data will be permanently removed.

---

### Analysis Methods

#### `clusters()`

Group similar documents into clusters based on term overlap.

**Parameters:**
- None

**Returns:**
- `List[List[int]]`: List of clusters, where each cluster is a list of document IDs that share similar content and vocabulary.

**Example:**
```python
document_clusters = drag.clusters()

for i, cluster in enumerate(document_clusters):
    print(f"Cluster {i}: {len(cluster)} documents")
    print(f"Document IDs: {cluster}")
```

**Note:** Uses a Markov chain approach with term overlap matrices to identify communities of related documents. Useful for discovering document themes and organizing large document collections.

---

#### `summarize(document_id, top_k=10)`

Generate an extractive summary of a document using TextRank algorithm.

**Parameters:**
- `document_id` (str): Unique identifier of the document to summarize.
- `top_k` (int, optional): Number of sentences to include in the summary. Default: 10

**Returns:**
- `List[str]`: List of the most important sentences from the document, ranked by their TextRank scores.

**Example:**
```python
summary = drag.summarize("doc_12345", top_k=5)

print("Document Summary:")
for sentence in summary:
    print(f"- {sentence}")
```

**Note:** TextRank scores sentences based on their structural importance within the document, similar to PageRank for web pages.

---

### Query Methods

#### `query_document(query, document_id, max_sentences=10)`

Generate a query-focused summary from a specific document.

**Parameters:**
- `query` (str): The search query or question to answer.
- `document_id` (str): Unique identifier of the document to query.
- `max_sentences` (int, optional): Maximum number of sentences to return. Default: 10

**Returns:**
- `List[str]`: List of sentences most relevant to the query, combining semantic similarity, keyword matching, and document structure.

**Example:**
```python
relevant_sentences = drag.query_document(
    query="What are the main findings about climate change?",
    document_id="doc_12345",
    max_sentences=5
)

print("Relevant passages:")
for sentence in relevant_sentences:
    print(f"- {sentence}")
```

**Note:** Uses a weighted combination of three scoring approaches:
- **Semantic similarity (alpha)**: Vector embeddings to capture meaning
- **Keyword matching (beta)**: Direct term overlap with query
- **Document structure (gamma)**: TextRank importance scores

---

#### `query(query, max_sentences=10, max_documens=3)`

Search across all documents and return ranked query-focused summaries.

**Parameters:**
- `query` (str): The search query or question to answer.
- `max_sentences` (int, optional): Maximum sentences per document summary. Default: 10
- `max_documens` (int, optional): Maximum number of documents to return. Default: 3

**Returns:**
- `List[dict]`: List of document results, where each result contains:
  - `document_id`: Unique identifier of the document
  - `answer`: List of relevant sentences from that document
  
  Results are ranked by relevance using a reranker model.

**Example:**
```python
results = drag.query(
    query="What are the benefits of renewable energy?",
    max_sentences=8,
    max_documens=5
)

for result in results:
    print(f"\nDocument ID: {result['document_id']}")
    print("Relevant passages:")
    for sentence in result['answer']:
        print(f"  - {sentence}")
```

**Note:** Performs a two-stage retrieval:
1. Initial retrieval using vector similarity and keyword search
2. Reranking using a cross-encoder model for optimal relevance

---

## Complete Example

```python
from DRAG import DRAG

# Initialize
drag = DRAG(
    db_name="research_papers.db",
    db_path="/data/databases",
    alpha=0.6,  # Prioritize semantic understanding
    beta=0.3,   # Some keyword matching
    gamma=0.1   # Light structural importance
)

# Upload documents
papers = [
    {"text": "Climate change paper content...", "source": "climate_2024.pdf"},
    {"text": "Renewable energy paper content...", "source": "renewable_2024.pdf"},
]

doc_ids = []
for paper in papers:
    doc_id = drag.upload(
        text=paper["text"],
        source=paper["source"],
        metadata={"year": 2024, "field": "Environmental Science"}
    )
    doc_ids.append(doc_id)
    print(f"Uploaded: {paper['source']} -> {doc_id}")

# Query across all documents
results = drag.query(
    query="How does renewable energy help combat climate change?",
    max_sentences=5,
    max_documens=3
)

print("\nSearch Results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Document: {result['document_id']}")
    for sentence in result['answer']:
        print(f"   - {sentence}")

# Get document clusters
clusters = drag.clusters()
print(f"\nFound {len(clusters)} document clusters")

# Generate summary of a specific document
summary = drag.summarize(doc_ids[0], top_k=3)
print("\nDocument Summary:")
for sentence in summary:
    print(f"- {sentence}")
```

## Advanced Configuration

### Tuning Weights

The `alpha`, `beta`, and `gamma` parameters control how different signals are weighted in query matching:

- **High alpha (0.7+)**: Emphasizes semantic meaning, good for conceptual queries
- **High beta (0.5+)**: Emphasizes exact keyword matches, good for technical/specific terms
- **High gamma (0.3+)**: Emphasizes document structure, good when key information appears in important sentences

### Recommended Presets

```python
# Semantic-focused (research, conceptual queries)
drag = DRAG(db_name="db.db", db_path="/data", alpha=0.7, beta=0.2, gamma=0.1)

# Keyword-focused (technical documentation, code search)
drag = DRAG(db_name="db.db", db_path="/data", alpha=0.3, beta=0.6, gamma=0.1)

# Balanced (general purpose)
drag = DRAG(db_name="db.db", db_path="/data", alpha=0.55, beta=0.35, gamma=0.10)
```

## License

Apache License 2.0
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
