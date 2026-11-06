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
