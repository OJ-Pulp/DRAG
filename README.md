# DRAG: Document-based Retrieval-Augmented Generation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

DRAG (Document-based Retrieval-Augmented Generation) is an extractive-style Retrieval-Augmented Generation (RAG) framework designed to improve coherence, relevance, and semantic fidelity over traditional chunk-based extractive methods in a document setting. Unlike conventional RAG approaches that split documents into chunks, DRAG performs **query-based summarization at the document level**, leveraging **graphical representations** and **graph-theoretic reasoning** to generate responses.

---

## Cite This Work

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
---

## Installation

1. Clone the repository:

```bash
pip install git+https://github.com/OJ-Pulp/DRAG.git
```

---

## Example
document_text = "The snow queen is ... the end."
document_name = "The Snow Queen by Eileen Kernaghan"

db_path = "./"
db_name = "books.sqlite"

drag = DRAG(db_name, db_path)
drag.upload(document_text, document_name)

question = "Who is the snow queen?"
answer = ' '.join(best_rag.query(question, max_sentences=7)[0]['answer'])
print(f"Answer: {answer}")

"""
Answer: Eileen Kernaghans *The Snow Queen* (2000) is a young-adult fantasy novel that reimagines Hans Christian Andersens 1844 fairy tale through a lens shaped by Scandinavian shamanism, Finnish mythology, and contemporary feminist thought. Set against the stark beauty of nineteenth-century northern Europe, the novel follows Gerda, a young Danish woman, and Ritva, a Smi shamans daughter, as they journey into the Arctic to confront Madame Aurore, a magician known as the Snow Queen, and rescue Gerdas childhood friend Kai. They learn that Aurore is no mere academic but the Snow Queen herself, a magician who rules the northern lands. Refusing to submit, Ritva enchants the palaces inhabitants into sleep, and the women flee with Kai, pursued by the Snow Queen until they are rescued by a southbound vessel. She described *The Snow Queen* as a feminist retelling, one that challenges Andersens Christian allegory of love and faith triumphing over reason. Published by Thistledown Press in 2000, *The Snow Queen* received largely positive reviews and won the 2001 Aurora Award for Best Novel. Ultimately, *The Snow Queen* stands as a deliberate reworking rather than a simple adaptation.
"""

___

## Documentation


