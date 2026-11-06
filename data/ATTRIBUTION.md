# Data sources, attribution and license scope

## CC BY-SA 4.0 material

The following distributed files contain Wikipedia text, sampled SQuAD questions/answers,
or a mixture of those sources and original educational examples. They are provided
under **Creative Commons Attribution-ShareAlike 4.0 International**, including the
project's original contributions to these mixed datasets:

- `data/raw/pretrain_en.txt`
- `data/raw/pretrain_he.txt`
- `data/raw/domain_notes.txt`
- `data/raw/sft_dialogues.jsonl`
- `data/raw/sft_dialogues_boosted.jsonl`
- `data/raw/sft_builder_base.jsonl`

The full license is [included here](../licenses/CC-BY-SA-4.0.txt).
Preserve attribution, link to the license, identify your changes, and license
adaptations of this material under CC BY-SA 4.0 or an allowed compatible license.
The root MIT License does not replace these terms.

### Wikipedia

Credit: **English and Hebrew Wikipedia contributors**. Individual article links
are listed in [the source manifest](raw/source_manifest.json); their history pages
identify contributors. The manifest includes the downloader's complete requested
article inventory, including titles that may redirect or have returned no extract.
Historical corpus files did not retain exact downloaded revision IDs. Do not
interpret the linked articles' current revisions as the exact historical snapshots.

Changes: plain-text extraction, whitespace normalization, concatenation, topic
selection, first-sentence truncation and conversion into educational Q&A/JSON rows.
No endorsement by Wikipedia or the Wikimedia Foundation is implied.
See [Wikimedia's reuse terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content).
Future downloads retain the resolved article URL and revision ID when available.

### SQuAD 2.0

Credit: **Pranav Rajpurkar, Robin Jia, Percy Liang and the SQuAD contributors**.
Source and license declaration: https://rajpurkar.github.io/SQuAD-explorer/.
Input dataset: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json.
The dataset's underlying passages originate from Wikipedia.

Changes: deterministic sampling of answerable questions, answer-length filtering,
text normalization, first-answer selection and conversion into system/user/assistant
dialogues. Mixed SFT files also include original educational examples and generated
variations. They are not official SQuAD releases or benchmark results.

## Original examples

`data/examples/` and the synthetic examples under `builder_projects/` use the root
MIT License. Contact addresses use reserved example domains. These fixtures contain
illustrative policy notes and synthetic flight data, not a licensed third-party
business dataset. The original scope prompt is project documentation, not training
data that must be imported automatically.

## Your own data and model outputs

Adding a URL or uploading a file does not establish permission to train on or
redistribute its content. Use data you control or have permission to use, record
its provenance and avoid private/confidential information. The licenses here
describe the supplied source files; they do not grant rights in your future uploads
or automatically determine the license of trained weights.
