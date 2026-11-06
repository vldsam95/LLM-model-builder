# Third-party notices

Original software and project documentation are offered under the root MIT License.
That grant covers only rights held by this project's contributors. Third-party
datasets and libraries retain their respective licenses.

## Data

The distributed Wikipedia/SQuAD-derived corpora and mixed SFT datasets listed in
[data/ATTRIBUTION.md](data/ATTRIBUTION.md) are CC BY-SA 4.0. Their contributors retain
copyright. See the full [CC BY-SA 4.0 license](licenses/CC-BY-SA-4.0.txt), source links,
description of transformations and [machine-readable provenance](data/raw/source_manifest.json).
The code's MIT grant does not relicense this content.

The small original samples in `data/examples/` and synthetic project examples in
`builder_projects/` are under MIT, except where an individual notice specifies otherwise.
Names, companies, policy descriptions, contact fields and flight rows in these examples
are illustrative; they are not represented as records of actual customers or employees.

## Models and tokenizers

No pre-trained checkpoints, tokenizer binaries or serialized predictive models are
included in this source release. The historical artifacts did not have complete
artifact-specific provenance/model cards and have been excluded from the publication
history. Their omission is a publication choice, not a claim that every trained model
automatically inherits the copyright license of its training data.

Training commands create local outputs. Before sharing your own outputs, review the
actual training sources, permitted uses, privacy/memorization risks and appropriate
artifact license, and provide a model card. This repository does not certify those outputs.

## Software dependencies

Dependencies are installed from their upstream distributions, not vendored here:
PyTorch, Flask, joblib, NumPy, pandas, openpyxl, PyYAML, requests, scikit-learn,
SentencePiece and tqdm. Their upstream license files remain part of those packages.
`requirements.txt` and `scripts/install_env.sh` identify the direct versions used.
This source release does not distribute their wheels, a virtual environment or a
container image. If you redistribute a bundled environment, include notices for
the exact direct/transitive dependencies and platform libraries in that distribution.
