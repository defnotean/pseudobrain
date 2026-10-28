# Released PseudoBrain Checkpoints on Hugging Face

All model weights are hosted on Hugging Face Hub under [`defnotean1`](https://huggingface.co/defnotean1). No large binary weights are stored in Git.

| Model | Architecture | HF Repository (NVFP4 + BF16 GDN) | Slim GDN-Only Shard | Model Card |
| :--- | :--- | :--- | :--- | :--- |
| **PseudoBrain-Nex-35B-A3B (`pb27-nexc`)** | `PB27MoEForConditionalGeneration` (`35B` total / `~3B` active, `30` native GDN + `10` hybrid `16K` SWA + GDN layers) | [`defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4`](https://huggingface.co/defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4) (`23 GiB`) | [`defnotean1/pb27-checkpoints/tree/main/nexc`](https://huggingface.co/defnotean1/pb27-checkpoints/tree/main/nexc) (`644 MB`) | [`PseudoBrain-Nex-35B-A3B-NVFP4/README.md`](PseudoBrain-Nex-35B-A3B-NVFP4/README.md) |
| **PseudoBrain-27B (`pb27c`)** | `PB27ForConditionalGeneration` (`27B` dense, `48` native GDN + `16` hybrid `16K` SWA + GDN layers) | [`defnotean1/PseudoBrain-27B-NVFP4`](https://huggingface.co/defnotean1/PseudoBrain-27B-NVFP4) (`20 GiB`) | [`defnotean1/pb27-checkpoints/tree/main/pb27c`](https://huggingface.co/defnotean1/pb27-checkpoints/tree/main/pb27c) (`3.5 GB`) | [`PseudoBrain-27B-NVFP4/README.md`](PseudoBrain-27B-NVFP4/README.md) |
