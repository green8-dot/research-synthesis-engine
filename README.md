# Research Synthesis Engine

Web scraping and NLP processing for research intelligence.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scrapy](https://img.shields.io/badge/Scrapy-2.11-green.svg)](https://scrapy.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7-blue.svg)](https://spacy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph-orange)](https://networkx.org/)

## Features

- Multi-source web scraping (Scrapy spiders)
- NLP text processing (spaCy, transformers)
- Knowledge graph construction (NetworkX)
- Report generation
- Research entity extraction

## Installation

```bash
git clone https://github.com/green8-dot/research-synthesis-engine
cd research-synthesis-engine
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

## Structure

```
src/
├── scrapers/          # Web scraping spiders
├── nlp/              # Natural language processing
├── knowledge_graph/  # Graph construction
├── pipeline/         # Data processing pipeline
├── models/           # Data models
├── api/              # API interfaces
└── utils/            # Utility functions
```

## Requirements

- Python 3.8+
- Dependencies in requirements.txt

## License

MIT License - See LICENSE file

## Author

Derek Green
- GitHub: [@green8-dot](https://github.com/green8-dot)
- LinkedIn: [derek-green-44723323a](https://www.linkedin.com/in/derek-green-44723323a/)
