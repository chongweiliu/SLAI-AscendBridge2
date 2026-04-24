# microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

- Date: 2026-04-23
- Outcome: stage-1 repaired, stage-3 completed

Key fix:
- `open_clip` with `hf-hub:` was still issuing HF HEAD requests even after weights had been downloaded into the adaptation-local cache. This made the model look permanently blocked on network.
- The stable repair was to stop using `hf-hub:` entirely and synthesize a `local-dir:` model folder inside the adaptation:
  - symlink `open_clip_pytorch_model.bin` from the adaptation-local snapshot
  - symlink BiomedBERT `config.json` / `tokenizer_config.json` / `vocab.txt` from local HF cache
  - rewrite `open_clip_config.json` so `text_cfg.hf_model_name` and `hf_tokenizer_name` both point to the local folder

Benchmark / optimization contract:
- baseline: `accuracy_run.py --use-pretrained --max-samples 50`
- perf: runtime-only `warmup(3x) + TASK_QUEUE_ENABLE=1`
- selected NPU: `13`
- output type: `clip_image_embeddings`
- dataset: `cifar100`

Observed result:
- baseline wall clock: `0.718996s`
- perf wall clock: `0.685599s`
- speedup ratio: `1.048712x`
- cosine similarity: `1.0`
- num_samples: `50`

Takeaway:
- For `open_clip` HF exports that wrap an HF text encoder, “weights downloaded” is not enough. If the config still names a remote `hf_model_name`, the text tower can keep reaching the network. Localize both the CLIP checkpoint and the text-encoder tokenizer/config before concluding the model is blocked.
