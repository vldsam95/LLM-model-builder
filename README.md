# Tiny Bilingual Assistant

Original code and documentation: [MIT License](LICENSE), copyright 2026
[vldsam95 and contributors](https://github.com/vldsam95/LLM-model-builder).
You may use, modify and redistribute the code while retaining its copyright and
permission notice. Suggestions and pull requests are welcome; see [Contributing](CONTRIBUTING.md).

Training data is licensed separately: Wikipedia/SQuAD-derived text and mixed SFT
files use **CC BY-SA 4.0**, with [attribution and scope](data/ATTRIBUTION.md).
See [third-party notices](THIRD_PARTY_NOTICES.md). Educational use does not waive
the original contributors' license terms.

This source release includes **no trained checkpoints, tokenizer binaries or
serialized predictive models**. Train them locally using the steps below or use
the builder's From Scratch Tiny mode.

Tiny Bilingual Assistant is a small educational GPT-style bilingual assistant built from scratch in PyTorch for CPU-only machines. It demonstrates the full mini-LLM pipeline end-to-end:

- SentencePiece tokenizer training
- small bilingual pretraining corpus preparation
- optional domain adaptation
- supervised fine-tuning with correct assistant-only loss masking
- local inference
- browser-based chat UI

This project is intentionally small, honest, and demo-oriented. It is meant to show understanding of the pipeline, not to compete with modern large instruction-tuned models.

## What It Is

- A tiny decoder-only transformer written from scratch in PyTorch
- A reusable educational framework for building small specialized assistants
- A bilingual English/Hebrew demo focused on:
  - short dialogue replies
  - simple translation
  - short definitions
  - basic marketing / SEO / AI help
  - lightweight JSON-style answers

## What It Is Not

- Not a production chatbot
- Not a general-purpose assistant
- Not a retrieval system
- Not a GPU/distributed training stack
- Not a Hugging Face Trainer workflow

## Hardware Assumptions

Example CPU-only development profile (adjust for your hardware):

- `8 vCPU`
- `23 GiB RAM`
- `~189 GB free disk`
- Linux VM on `KVM/QEMU`
- CPU-only, no CUDA / NVIDIA GPU

The safe default config is about `5.6M` parameters. An optional larger config is included for experimentation and pushes closer to the upper single-digit millions.

## Architecture

Model:

- token embeddings
- learned positional embeddings
- masked multi-head self-attention
- residual connections
- LayerNorm or RMSNorm
- GELU FFN or SwiGLU
- optional grouped-query style KV sharing via `n_kv_heads`
- tied input/output embeddings

Training curriculum:

1. Download / curate small bilingual corpora
2. Train a SentencePiece tokenizer
3. Pretrain on compact bilingual text
4. Optionally run domain adaptation on SEO / AI / marketing notes
5. Fine-tune on dialogue SFT data
6. Compare checkpoints on a fixed prompt set
7. Run the local web UI

## Data Sources

The default small corpus is built from lightweight public sources:

- Wikipedia extracts through the official MediaWiki API
  - docs: `https://www.mediawiki.org/wiki/API:Main_page`
- SQuAD v2.0 from the official project page
  - site: `https://rajpurkar.github.io/SQuAD-explorer/`

The project also adds curated hand-written bilingual SFT examples for translation, greetings, marketing prompts, and JSON-style outputs.

Important:

- Use only data you are allowed to use.
- Preserve the Wikipedia article credits and transformation notes in `data/ATTRIBUTION.md` and `data/raw/source_manifest.json`.
- SQuAD 2.0 is CC BY-SA 4.0; preserve its credit and license when distributing derived datasets.

## Project Layout

```text
tiny-bilingual-assistant/
  README.md
  requirements.txt
  configs/
  data/
    raw/
    processed/
    examples/
  tokenizer/
  src/
    model/
    data/
    training/
    inference/
    web/
    utils/
  scripts/
  checkpoints/
  logs/
```

## Setup

System dependencies were installed with:

```bash
apt-get update
apt-get install -y python3-pip python3-venv python3-dev build-essential curl wget git jq unzip
```

Create and populate the virtual environment:

```bash
cd /path/to/LLM-model-builder
./scripts/install_env.sh
```

This installs the CPU-only PyTorch wheel from the official PyTorch CPU index, then installs the rest of the project requirements. This avoids accidentally pulling a CUDA-heavy build on a CPU-only VM.

## End-to-End Commands

Download raw data:

```bash
source .venv/bin/activate
python scripts/download_data.py --raw-dir data/raw
```

Validate raw data:

```bash
python scripts/validate_data.py --raw-dir data/raw
```

Prepare tokenizer and processed tensors:

```bash
python scripts/prepare_data.py --download --raw-dir data/raw --processed-dir data/processed --vocab-size 3200 --sequence-length 192
```

Run pretraining:

```bash
./scripts/run_pretrain.sh
```

Optional domain adaptation:

```bash
./scripts/run_domain.sh
```

Run SFT:

```bash
./scripts/run_sft.sh
```

If you want to skip domain adaptation, point `init_checkpoint` in [configs/sft_cpu.yaml](configs/sft_cpu.yaml) to `checkpoints/tiny_cpu_pretrain/best.pt` instead of the domain checkpoint.

For the most presentation-friendly local demo on this CPU-only VM, use the tuned demo config:

```bash
python -m src.training.train_sft --config configs/sft_demo_cpu.yaml
```

Compare checkpoints:

```bash
python -m src.training.eval \
  --tokenizer tokenizer/tiny_bilingual.model \
  --checkpoints \
    checkpoints/tiny_cpu_pretrain/best.pt \
    checkpoints/domain_cpu_adapt/best.pt \
    checkpoints/tiny_cpu_builder_base/best.pt \
    checkpoints/tiny_cpu_sft_demo_boosted/best.pt
```

Run the web UI:

```bash
./scripts/run_web.sh
```

Then open `http://127.0.0.1:7860` in a browser.

## Source release and first model

The repository includes source code, configuration, attributed training corpora,
original samples and synthetic builder examples. Virtual environments, processed
tensors, runtime state, trained weights, serialized predictive models and tokenizer
binaries are local generated outputs and are not included in Git.

To create the neutral base used by Fast Adapter projects:

```bash
./scripts/install_env.sh
source .venv/bin/activate
python scripts/prepare_data.py --download --sequence-length 128
./scripts/run_pretrain.sh
python scripts/prepare_builder_base.py
python -m src.training.train_sft --config configs/builder_base_sft_cpu.yaml
./scripts/run_web.sh
```

Downloading data contacts Wikipedia and the official SQuAD source. Keep the
attribution manifest with any redistributed dataset. Training can take significant
CPU time; these commands do not claim a particular model quality or benchmark.
The existing attributed raw corpora may also be processed without `--download`.
That offline path cannot restore the original per-article JSONL metadata; the
bundled historical attribution remains in `data/raw/source_manifest.json`.

Open http://127.0.0.1:7860. The educational web UI has no application authentication
and supports local file imports and training. It defaults to loopback; see
[local deployment notes](infra/README.md) before changing the bind address.

Existing historical training results describe earlier private experiments; they
are not validation results for distributed model weights, because no such weights
are supplied in this source release. Review provenance, privacy and licensing again
before sharing models that you train yourself.

## Model Builder

The web app now also includes a reusable builder at `/builder`.

It now supports two families of projects:

- `Language Model`: tiny GPT-style assistants and specialized text models
- `Predictive AI`: CPU-friendly tabular models for classification, regression, and simple forecasting

For language-model projects, it is designed to help you create new tiny text assistants on top of the same pipeline, without inheriting the SEO / marketing "personality" of the original demo:

- create a named project with specialization, topics, and system prompt
- paste raw text, notes, or FAQs
- import visible page text from URLs
- upload text files or JSONL dialogue files
- choose a profile:
  - `General Tiny Assistant`
  - `Extraction / JSON Schema Builder`
  - `FAQ / Internal Knowledge Assistant`
  - `Classification / Scoring Mini-Model`
- choose a training mode:
  - `Fast Adapter`: fine-tune on top of an existing checkpoint
  - `From Scratch Tiny`: project tokenizer + pretrain + domain + SFT
- choose a foundation mode:
  - `Project Only`: only use the project corpus
  - `General Open Pack`: mix the project corpus with the local open foundation pack
- choose a model size preset for `From Scratch Tiny`:
  - `~5.6M Safe CPU`
  - `~11.2M Balanced CPU`
  - `~15.2M Stretch CPU`
- launch training jobs from the browser
- inspect logs and open a project-specific preview chat
- browse a central local checkpoint registry in the UI

For predictive AI projects, the builder can also:

- upload a `CSV` or `XLSX` dataset
- define `target column`, optional `time column`, and feature columns
- choose `classification`, `regression`, or `forecasting`
- train a local tabular model on CPU
- generate validation metrics, feature importance, and a visual dashboard
- expose a simple browser-based prediction playground for non-technical users

Important realism note:

- text assistants and predictive tabular models are realistic on this CPU-only machine
- predictive tabular models are usually simpler, faster, and more practical than tiny LLMs for numeric business problems
- this is still not a universal GUI for every type of AI model
- vision, audio, and larger deep-learning forecasting stacks would still need different pipelines

### Builder Modes

`Fast Adapter`

- starts from a neutral base checkpoint
- is the recommended path when you want a new narrow assistant quickly
- works best when you already have notes, FAQs, policies, or a few structured examples

`From Scratch Tiny`

- trains a project tokenizer and a new tiny GPT-style model from scratch
- can optionally bootstrap from the local `General Open Pack`
- is slower, but is the correct route when you want a new tiny model family instead of only adapting an existing one

`Predictive AI`

- does not use tokenizer / pretraining / SFT
- is meant for structured rows from spreadsheets or exports
- is the right path for tasks such as delay prediction, lead scoring, demand regression, or simple time forecasting
- uses local `pandas + scikit-learn` pipelines with browser-based metrics and prediction UI

### Foundation Sources

The local `General Open Pack` currently uses:

- `data/raw/pretrain_en.txt`
- `data/raw/pretrain_he.txt`
- `data/raw/sft_builder_base.jsonl`

These are the builder's generic foundation files. They are meant to provide general bilingual language prior and neutral assistant behavior rather than a domain-specific personality.

### Registry vs Ollama

The builder now includes a central checkpoint registry in the UI. This is similar to an internal local model catalog:

- it can list built-in checkpoints
- it can list project checkpoints created through the builder
- it can use those checkpoints as new adapter bases

It is not an Ollama-style runtime container manager. It does not currently provide a unified pull / tag / serve daemon for arbitrary third-party GGUF models.

## Predictive AI Guidance

If your goal is to predict a number, a label, or a short future horizon from spreadsheet data, a predictive tabular model is usually simpler than an LLM:

- it needs less data preparation
- it trains faster on CPU
- it is easier to evaluate honestly with metrics such as `MAE`, `RMSE`, `R2`, `Accuracy`, or `F1`
- it usually gives more stable results for operational forecasting problems

Example use cases:

- flight delay prediction
- lead scoring
- support ticket priority classification
- churn / risk scoring
- simple demand or volume forecasting

### Practical CPU Sizing Guidance

On this VPS, the practical latency / quality tradeoff looks like this:

- `~5M-6M` params: best default for interactive demos and responsive replies
- `~10M-12M` params: the best balanced ceiling for this machine if you still want reasonable reply speed
- `~15M+` params: feasible for experiments, but noticeably slower for live chat

If training time is not the constraint but interactive speed matters, the sweet spot on this server is usually the `5M-12M` range.

## UI Features

- chat window with message history
- clear / regenerate buttons
- checkpoint selector
- system instruction box
- temperature / top-k / max token controls
- JSON-style answer toggle
- demo presets:
  - Helpful Assistant
  - SEO Assistant
  - Hebrew Tutor
  - Translation Mode
  - JSON Output Mode

The frontend supports Hebrew RTL rendering.

## SFT Masking Logic

Dialogue records are stored as JSONL:

```json
{"system":"...", "user":"...", "assistant":"..."}
```

The formatter injects special tokens:

- `<sys>`
- `<user>`
- `<assistant>`

Loss is masked with `-100` for:

- system tokens
- user tokens
- padding
- the `<assistant>` marker itself

Loss is computed only on assistant response tokens. This is the key behavior that makes SFT training explicit and educational instead of treating the whole prompt as a target.

## Specialization Choice

This demo targets a narrow specialization because tiny models overfit quickly and do not generalize like large instruction-tuned systems. The chosen specialization fits the hardware and the data size:

- definitions
- translation
- tiny marketing/SEO outputs
- short bilingual assistant replies
- optional compact JSON-like replies

## Tokenizer Notes

One shared tokenizer is used for English and Hebrew. For a tiny model, vocabulary size is a tradeoff:

- too small: worse segmentation and more fragments
- too large: more embedding parameters and weaker statistics per token

`3200` is a safe educational default here. It is small enough for a CPU-friendly model and large enough to support mixed English/Hebrew text reasonably.

## Configs

- [configs/tiny_cpu.yaml](configs/tiny_cpu.yaml): safe default pretraining config
- [configs/domain_cpu.yaml](configs/domain_cpu.yaml): optional domain adaptation stage
- [configs/sft_cpu.yaml](configs/sft_cpu.yaml): SFT config
- [configs/sft_demo_cpu.yaml](configs/sft_demo_cpu.yaml): presentation-oriented SFT config tuned for the current CPU VM
- [configs/small_cpu.yaml](configs/small_cpu.yaml): larger experimental config

## Example Prompts

Use prompts such as:

- `Hello`
- `What is SEO?`
- `What is AI?`
- `מה זה SEO?`
- `מה זה בינה מלאכותית?`
- `Translate 'house' to Hebrew`
- `תתרגם לאנגלית: מחשב`
- `Write a short meta description for an AI article`
- `ענה לי ב-JSON`

## Limitations

- short context windows only
- weak world knowledge outside the small corpus
- limited translation quality
- weak multi-turn consistency
- system prompt acts only as a soft conditioning signal
- JSON formatting is learned behavior, not robust constrained decoding

## Demo Guardrail

The `ChatEngine` uses a pragmatic local fallback on the SFT examples when the tiny model returns an empty or obviously degenerate output. This keeps the browser demo stable for managerial presentation while remaining honest about the model's size and limits.

## Recommended Next Improvements

- add more licensed bilingual domain text
- expand Hebrew SFT coverage
- add small prompt-set regression tests
- add simple perplexity reporting
- add streaming generation in the UI
- add a second specialization by reusing the same framework

## Original Creation Prompt

```text
Build me a complete, runnable, cleanly organized end-to-end codebase from scratch for a tiny GPT-style bilingual dialogue model, designed primarily for educational value, internal demo value, and managerial presentation, with realistic constraints for CPU-only local/server execution.

High-level product goal
This project should result in a tiny bilingual educational AI assistant with a narrow, realistic specialization:
- short dialogue responses
- simple English/Hebrew translation
- short definitions and factual mini-explanations
- basic marketing / SEO / AI-oriented assistance
- optional simple JSON-style responses for constrained prompts

This is not intended to be a general chatbot. It should feel like a tiny specialized assistant that demonstrates the full LLM pipeline from scratch.

Important context and constraints
- This project is intentionally educational and demonstrational, not production-grade.
- The target machine is CPU-only, with no CUDA GPU available.
- Server specs:
  - 8 vCPU
  - 23 GiB RAM
  - ~189 GB free disk
  - Linux VM on KVM/QEMU
  - No dedicated NVIDIA GPU, no nvidia-smi
- Therefore, do NOT design this around a 10–30M parameter model or a consumer GPU workflow.
- Instead, adapt the architecture, training process, dataset sizes, defaults, and UX to this environment so the project is actually feasible, understandable, and runnable end-to-end.
- The goal is to build something genuinely from scratch in PyTorch that demonstrates understanding of the full LLM pipeline:
  - tokenizer training
  - data preparation
  - causal decoder-only transformer
  - next-token pretraining
  - supervised fine-tuning for dialogue
  - inference
  - simple browser-based UI
- The model should feel like a tiny AI dialogue assistant, even if limited.
- The model should support English and Hebrew, with realistic expectations and honest limitations.
- This should be optimized for successful completion over a weekend or a few focused work sessions, with minimal dependencies and clean educational code.

Product specialization and expected end behavior
The final model should behave like a tiny bilingual "Marketing, Knowledge, and Translation Assistant".

Expected capabilities after SFT:
- answer short prompts such as:
  - "Hello"
  - "What is SEO?"
  - "What is AI?"
  - "מה זה SEO?"
  - "מה זה בינה מלאכותית?"
- perform simple bilingual translation such as:
  - "Translate 'house' to Hebrew"
  - "תתרגם לאנגלית: מחשב"
- generate short, simple marketing-style outputs such as:
  - a short article title
  - a short meta description
  - a one-line explanation
- optionally produce short JSON-style responses for constrained prompts such as:
  - "Explain SEO in JSON"
  - "ענה לי ב-JSON"

The model does NOT need:
- long reasoning
- coding assistance
- advanced world knowledge
- long multi-turn coherence
- complex tool use
- retrieval
- enterprise deployment

Core project requirements
Create a complete codebase that includes:

1. Model scope and realistic size
- Build a tiny decoder-only GPT-style language model in PyTorch.
- Target a realistic CPU-friendly size, preferably around 3M–10M parameters, with the default config closer to the safer end of that range for CPU-only execution.
- The implementation must be from scratch, not using Hugging Face Trainer or prebuilt GPT model classes.
- Use:
  - token embeddings
  - learned positional embeddings
  - masked multi-head self-attention
  - LayerNorm
  - MLP/feed-forward blocks
  - residual connections
  - final LM head
- Keep the implementation simple, readable, and educational.

2. Tokenizer
- Train a SentencePiece tokenizer from project data.
- It must support both English and Hebrew in one tokenizer.
- Include special tokens needed for dialogue and training, for example:
  - <pad>
  - <bos>
  - <eos>
  - <unk>
  - <sys>
  - <user>
  - <assistant>
- Choose a realistic vocabulary size for a tiny bilingual model, likely in the low-thousands range.
- Add a clear explanation in the README of how tokenizer size and bilingual support affect model quality.

3. Data pipeline
Design the codebase around a two-stage data pipeline:

A. Pretraining data
- Small general-language corpus for next-token pretraining.
- Support English and Hebrew text corpora.
- Use plain text / markdown / txt inputs.
- Include scripts for cleaning and concatenating source files.

B. SFT dialogue data
- Lightweight supervised fine-tuning on dialogue-formatted examples.
- The model should learn a consistent assistant format and produce short dialogue-style responses.
- Support bilingual dialogue examples in English and Hebrew.
- Include mixed-language examples where useful, such as:
  - translation mini-tasks
  - short definitions
  - short Q&A
  - greeting dialogues
  - short SEO / AI / marketing assistant-style responses

4. Recommended data sources and project treatment of them
Bake the following assumptions directly into the project structure and README:

Pretraining sources may include:
- small curated English Wikipedia-derived text
- small curated Hebrew Wikipedia-derived text
- optionally small cleaned markdown notes or plain text domain text related to:
  - SEO
  - digital marketing
  - AI basics
  - automation
  - websites / content / search engines

Dialogue/SFT sources may include:
- hand-written JSONL examples
- converted QA-style examples
- small subtitle/dialogue-inspired examples
- tiny translation examples
- tiny definition examples
- small structured assistant examples for:
  - SEO explanations
  - AI definitions
  - title generation
  - meta description generation
  - JSON-style outputs

Do NOT assume huge datasets.
The codebase should be built for small, curated datasets in the ~10MB–30MB total range, with smaller subsets also supported.

Include placeholder sample files and a clear schema for where the user will place:
- data/raw/pretrain_en.txt
- data/raw/pretrain_he.txt
- data/raw/domain_notes.txt
- data/raw/sft_dialogues.jsonl

Add scripts that:
- validate data formatting before training
- clean and normalize text
- build train/validation splits
- report basic dataset statistics

Add README guidance on data licensing considerations and explain that users should only use data they are permitted to use.

5. SFT design
Implement SFT in a correct but lightweight way:
- Dialogue examples should be stored in a structured format such as JSONL.
- Recommended schema:
  {
    "system": "...optional system instruction...",
    "user": "...",
    "assistant": "..."
  }
- During SFT, construct training text with a consistent template using special tokens.
- Very important: compute loss only on the assistant response tokens, while masking system/user prompt tokens with ignore_index = -100.
- This masking logic must be clearly implemented and documented.
- The system should support an optional system prompt field because the UI will expose it as a configuration area, but the system prompt behavior should remain simple and explicit, not over-engineered.

6. Dialogue behavior goals
The model is small and CPU-trained, so do not overclaim.
Design the project so that after SFT the model can:
- answer short dialogue prompts
- maintain a simple assistant format
- respond in English or Hebrew
- handle simple translation prompts
- handle short definitions and factual patterns seen in training
- produce short marketing-style outputs
- optionally answer in JSON-like structure for very simple constrained cases

The code and README must explicitly explain that this is:
- a tiny educational bilingual autoregressive language model
- a small specialized assistant demo
- not a production chatbot
- not a replacement for large instruction-tuned LLMs

7. JSON output support
Implement optional simple JSON-style output support in an intentionally lightweight way.
This does NOT need to be robust structured decoding.
Instead:
- include a small set of SFT examples where the assistant responds in compact JSON
- optionally include an inference mode that nudges output toward JSON via a formatting prefix or system prompt
- document limitations honestly

8. Training stages
Implement two clearly separated training workflows:

Stage 1: Pretraining
- causal next-token prediction on general text
- train/validation split
- periodic evaluation on validation loss
- checkpoint saving
- best-model saving
- logging of tokens/sec or samples/sec if practical

Stage 2: SFT
- fine-tune from the pretrained checkpoint
- lower learning rate than pretraining
- dialogue-formatted data
- loss masking on assistant tokens only
- validation split and checkpointing

9. CPU-friendly training adaptation
This is critical.
Adapt everything to CPU-only operation:
- no distributed training
- no CUDA assumptions
- no requirement for mixed precision if unavailable
- mixed precision should be optional and disabled automatically on CPU
- gradient accumulation can remain supported, but defaults should be realistic for CPU
- use small default batch sizes and sequence lengths
- include practical defaults such as sequence length 128 or 256
- include a “tiny safe default config” that is likely to run on this machine
- include a slightly larger optional config for experimentation
- training must be resumable from checkpoints
- do not optimize for maximal speed; optimize for clarity and successful completion

10. Overfitting mitigation
Because the datasets are intentionally small, include simple measures against overfitting:
- train/validation split
- dropout
- weight decay
- early stopping or patience-based stopping
- gradient clipping
- small LR schedule if appropriate
- clear README explanation of why tiny models on tiny datasets can overfit quickly

11. Inference
Implement local inference with:
- checkpoint loading
- top-k sampling
- temperature
- max new tokens
- stop token handling if appropriate
- language-agnostic prompt handling for English/Hebrew
- a clean generation wrapper

12. Browser-based graphical interface instead of CLI
Do NOT make the main user interaction CLI-only.
Instead, create a small local web application that runs on a local URL, such as localhost on a configurable port.

The UI should be simple, lightweight, and practical.

Main requirements:
- browser-based chat window
- message history area
- user input box
- send button
- clear chat button
- optional regenerate button if easy
- side panel or configuration section with:
  - model checkpoint selector
  - system prompt field
  - temperature
  - top-k
  - max new tokens
  - language hint or mode selector (optional)
  - toggle for “JSON-style answer mode” (optional)
  - preset buttons such as:
    - Helpful Assistant
    - SEO Assistant
    - Hebrew Tutor
    - Translation Mode
- clean display for assistant responses
- proper support for Hebrew RTL rendering in the frontend
- clear handling of multiline input
- local-only design is fine; no auth needed
- should be easy to run with a single command after setup

UI architecture guidance
- Keep the stack minimal.
- A simple Flask or FastAPI backend is fine.
- Frontend can be:
  - server-rendered templates with minimal JS, or
  - a very light static frontend
- Do not build an overcomplicated SPA.
- The goal is a working local demo UI, not a full product.
- Include README instructions for launching the web UI locally.

13. System prompt / configuration design
Because this is a tiny model, do not pretend the system prompt is as powerful as in large LLMs.
However, the UI should include a simple system prompt/configuration area because it is useful for demonstration.

Design it like this:
- optional text box labeled “System Instruction”
- if filled, inject it in the dialogue template using a system token/segment
- explain in README that for a tiny model this acts as a soft conditioning signal only
- support a few demo presets, such as:
  - Helpful Assistant
  - SEO Assistant
  - Hebrew Tutor
  - Translation Mode
  - JSON Output Mode

14. Project organization
Organize the project in a clean folder structure, for example:

project_root/
  README.md
  requirements.txt
  pyproject.toml (optional)
  .env.example (if needed)
  configs/
    tiny_cpu.yaml
    small_cpu.yaml
    sft_cpu.yaml
  data/
    raw/
    processed/
    examples/
  tokenizer/
    train_tokenizer.py
    tokenizer_utils.py
  src/
    model/
      config.py
      layers.py
      transformer.py
    data/
      preprocess.py
      datasets.py
      dialogue_format.py
      validators.py
    training/
      train_pretrain.py
      train_sft.py
      eval.py
      loops.py
      optim.py
      checkpointing.py
    inference/
      generate.py
      chat_engine.py
    web/
      app.py
      routes.py
      templates/
      static/
    utils/
      seed.py
      logging_utils.py
      config_loader.py
  scripts/
    prepare_data.py
    validate_data.py
    run_pretrain.sh
    run_sft.sh
    run_web.sh
  checkpoints/
  logs/

You may improve this structure, but keep it clean and simple.

15. Deliverables
The completed codebase must include:
- full runnable source code
- requirements.txt with minimal dependencies
- sample config files
- sample tiny datasets or dataset templates
- sample bilingual SFT JSONL examples, including English and Hebrew
- sample examples for the chosen specialization:
  - short SEO explanations
  - AI definitions
  - translation examples
  - title generation examples
  - meta description examples
  - JSON-style responses
- tokenizer training script
- pretraining script
- SFT training script
- evaluation script or validation reporting
- local web chat app
- checkpoint loading for inference
- concise but clear README
- execution steps from zero to running demo
- comments and explanations in code where educationally useful

16. README requirements
The README should be concise but solid and should include:
- what the project is
- what it is not
- hardware assumptions
- the chosen specialization and why it fits a tiny model
- how to prepare data
- how to train tokenizer
- how to run pretraining
- how to run SFT
- how to start the web UI
- how to test English and Hebrew prompts
- how to switch checkpoints
- how to use system prompts
- expected limitations
- recommended next improvements

17. Example UX behavior
After running the project, I want to be able to:
- open a local browser URL
- select a checkpoint
- optionally type a system instruction like:
  - "Answer briefly in Hebrew"
  - "Act as an SEO assistant"
  - "Respond in JSON"
- type:
  - "Hello"
  - "מה זה בינה מלאכותית?"
  - "Translate 'house' to Hebrew"
  - "Write a short meta description for an AI article"
  - "ענה לי ב-JSON"
- and receive short model-generated responses in the chat window

18. Engineering standards
- Write readable, production-style educational code
- Keep dependencies minimal
- Prefer clarity over cleverness
- Use type hints where practical
- Add basic error handling
- Make scripts deterministic when possible via seeds
- Keep the codebase runnable on Linux
- Do not hide complexity in giant utility files
- Separate concerns cleanly

19. Important realism constraints
Please do not overscope this project.
Do not design for:
- large-scale GPU training
- distributed training
- giant datasets
- advanced RLHF
- quantization pipelines
- LoRA
- Hugging Face Trainer abstractions
- sophisticated serving stacks
- enterprise authentication
- websocket-heavy frontend architecture
unless something extremely lightweight is justified

20. Desired output from you
Return:
1. the complete runnable codebase
2. all source files in full
3. sample configs
4. sample tiny bilingual SFT dataset
5. example pretraining data placeholders/instructions
6. clear setup and execution instructions
7. concise explanation of architecture and tradeoffs

21. Efficiency-oriented design philosophy
Take inspiration from the efficiency mindset seen in modern large-model systems such as DeepSeek, but adapt it realistically for a tiny CPU-only educational model.

Specifically:
- prioritize architectural simplicity and training efficiency over scale
- keep the default model small and safe for CPU
- support weight tying between token embeddings and LM head
- support a configurable normalization choice (LayerNorm baseline, optional RMSNorm)
- support a configurable feed-forward choice (GELU baseline, optional SwiGLU)
- optionally support Grouped-Query Attention (GQA) or Multi-Query Attention (MQA) as a lightweight inference-efficiency variant, but keep standard masked self-attention as the simplest baseline
- keep sequence lengths modest by default
- structure training as a small curriculum:
  1. tokenizer training and dataset validation
  2. general pretraining
  3. optional short domain adaptation on SEO/AI/marketing text
  4. supervised fine-tuning for dialogue
  5. evaluation on fixed bilingual prompts
- include a small evaluation harness comparing:
  - pretrained checkpoint
  - domain-adapted checkpoint (if implemented)
  - SFT checkpoint
- document all efficiency/performance tradeoffs clearly in the README

Important:
Do NOT introduce large-scale complexity such as MoE, MLA, distributed routing, advanced load balancing, or other hyperscale-only techniques. The goal is to borrow the efficiency mindset, not the hyperscale architecture.

22. Domain adaptation stage
In addition to general pretraining and SFT, optionally include a very small intermediate domain-adaptation stage on compact SEO/AI/marketing-related text before SFT. This stage should be lightweight, optional, and easy to skip, but supported by the project structure and configs.
```
