.PHONY: help install smoke convert export compose-nvfp4 serve serve-nex eval eval-1m eval-code

PYTHON ?= python3
MODEL_DIR ?= defnotean1/PseudoBrain-Nex-35B-A3B-NVFP4
SERVED_NAME ?= pb27-nex
PORT ?= 8000

help:
	@echo "PseudoBrain Release Makefile"
	@echo "  make install       Install pseudobrain core and pb27_vllm plugin in editable mode"
	@echo "  make smoke         Run CPU smoke test of pseudobrain conversion core & exact memory tier"
	@echo "  make serve         Serve $(MODEL_DIR) on port $(PORT) via serving/pb27_vllm/serve.sh"
	@echo "  make eval          Run 8K-200K long-context exact recall & multi-hop suite"
	@echo "  make eval-1m       Run 250K-1M extended long-context exact recall & multi-hop suite"
	@echo "  make eval-code     Run 33-problem x 2-sample medium-hard Python coding benchmark"
	@echo "  make convert       Run parallel Gated DeltaNet distillation on a teacher checkpoint"

install:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install --no-deps -e serving/pb27_vllm

smoke:
	$(PYTHON) -c "import torch, pseudobrain, efficiency.transplant; pseudobrain.EXACT_MEMORY.update({'enabled': True, 'blocks': 2, 'block_size': 8}); arc = pseudobrain.archive_append(None, torch.randn(1, 2, 32, 16), torch.randn(1, 2, 32, 16)); rk, rv = pseudobrain.retrieve(arc, torch.randn(1, 4, 2, 16), groups=2); assert rk.shape == (1, 2, 16, 16) and rv.shape == (1, 2, 16, 16); print('CPU smoke test passed:', rk.shape)"

serve:
	bash serving/pb27_vllm/serve.sh $(MODEL_DIR) $(SERVED_NAME) $(PORT)

eval:
	bash scripts/bench/custom_longctx_pertoken.sh $(SERVED_NAME) $(MODEL_DIR)

eval-1m:
	bash scripts/bench/custom_longctx_extended.sh $(SERVED_NAME) $(MODEL_DIR)

eval-code:
	$(PYTHON) scripts/bench/custom_code_bench.py --url http://127.0.0.1:$(PORT)/v1/chat/completions --model $(SERVED_NAME) --tier medium --n 2 --out reports/pb27/custom_code_eval

convert:
	$(PYTHON) -m scripts.convert.convert_token --help
