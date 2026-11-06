# Contributing

Issues, suggestions and pull requests are welcome. Please describe the problem,
the proposed change and how you tested it. Small, focused changes are easiest to review.

Original code contributions are submitted under the repository's MIT License.
Retain existing copyright and license notices. Do not submit secrets, personal data,
confidential documents, copied code without its required notices, or training data
whose redistribution rights you cannot establish.

For data changes, identify the source, rights holder, license, modifications and
applicable files in `data/ATTRIBUTION.md` and `data/raw/source_manifest.json`.
Wikipedia/SQuAD-derived data remains under CC BY-SA 4.0, not MIT.

Generated checkpoints, tokenizers, serialized models, uploads and runtime state
are local outputs. Do not commit them. To propose a model release, first provide
an artifact-specific model card, training-data provenance and a separate license review.

Run the educational smoke tests before submitting:

```bash
python -m unittest discover -s tests -p '*test*.py'
```

Credit the original project by preserving its MIT notice. A link to
https://github.com/vldsam95/LLM-model-builder in derivative documentation is appreciated.
