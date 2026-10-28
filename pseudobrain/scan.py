"""Input-gated, parallel-scan recurrence with constant-size inference state.

Inspired by minGRU (Feng et al., 2024, arXiv:2410.01201), not an exact reproduction.
Backends: 'reference' (autograd through a work-efficient tree of affine
transforms), 'parallel' (the same tree with an O(B*T*D) custom backward),
'compiled' (torch.compile of 'parallel') and 'fused' (efficiency/kernels/
scan_triton.py: one Triton launch per direction that also fuses the minGRU
gate math; falls back to 'parallel' off CUDA, without Triton, or for
non-fp32 state). No GRU checkpoint conversion is claimed.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.autograd.function import once_differentiable
from functools import lru_cache
from torch.utils.checkpoint import checkpoint


def affine_prefix(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive prefix of f_t(h)=a_t*h+b_t; [batch,time,width], autograd-safe.

    Pair-reduce/recurse/interleave has O(T) arithmetic work and O(log T) dependency
    depth. Floating-point reassociation is not bitwise equal to serial recurrence.
    No divisions, logarithms, or exponentially large rescaling factors are used.
    """
    if a.shape != b.shape or a.ndim != 3 or a.shape[1] < 1:
        raise ValueError('affine_prefix expects matching nonempty [B,T,D] tensors')
    n = a.shape[1]
    if n == 1:
        return a, b
    pairs = n // 2
    right_a = a[:, 1::2]
    pa, pb = affine_prefix(right_a * a[:, 0:2*pairs:2],
                           b[:, 1::2] + right_a * b[:, 0:2*pairs:2])
    even_a = torch.cat((a[:, :1], a[:, 2::2] * pa[:, :(n-1)//2]), dim=1)
    even_b = torch.cat((b[:, :1], b[:, 2::2] + a[:, 2::2] * pb[:, :(n-1)//2]), dim=1)
    out_a = torch.stack((even_a[:, :pairs], pa), dim=2).flatten(1, 2)
    out_b = torch.stack((even_b[:, :pairs], pb), dim=2).flatten(1, 2)
    if n % 2:
        out_a = torch.cat((out_a, even_a[:, -1:]), dim=1)
        out_b = torch.cat((out_b, even_b[:, -1:]), dim=1)
    return out_a, out_b


def _affine_values(a: torch.Tensor, b: torch.Tensor, initial: torch.Tensor):
    pa, pb = affine_prefix(a, b)
    return pa * initial[:, None] + pb


class _ParallelAffine(torch.autograd.Function):
    """First-order reverse-mode scan, without retaining the prefix-tree graph.

    dh[t] = grad_y[t] + a[t+1]*dh[t+1] is another affine scan in reverse.
    Saves only coefficients, returned states and the initial state, O(B*T*D).
    Higher derivatives are deliberately unsupported; use the reference backend.
    """
    @staticmethod
    def forward(ctx, a, b, initial):
        values = _affine_values(a, b, initial)
        ctx.save_for_backward(a, values, initial)
        return values

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_values):
        a, values, initial = ctx.saved_tensors
        # The final recurrence has no successor; a[last+1] is zero.
        reverse_a = torch.cat((a[:, 1:], torch.zeros_like(a[:, :1])), 1).flip(1)
        _, reverse_grad = affine_prefix(reverse_a, grad_values.flip(1))
        adjoint = reverse_grad.flip(1)
        previous = torch.cat((initial[:, None], values[:, :-1]), 1)
        return adjoint * previous, adjoint, adjoint[:, 0] * a[:, 0]


def parallel_affine(a: torch.Tensor, b: torch.Tensor, initial: torch.Tensor):
    """Exact recurrence in real arithmetic; floating reassociation has roundoff.

    O(T) work, O(log T) scan depth at fixed batch/width. A single recurrent
    inference update still does constant work with respect to past history.
    No Tensor Core use is asserted for the elementwise scan itself.
    """
    if (a.shape != b.shape or a.ndim != 3 or a.shape[1] < 1
            or initial.shape != (a.shape[0], a.shape[2])):
        raise ValueError('expected aligned nonempty [B,T,D] and [B,D]')
    if not a.is_floating_point() or a.dtype != b.dtype or a.dtype != initial.dtype:
        raise ValueError('scan inputs must have a common floating-point dtype')
    if a.device != b.device or a.device != initial.device:
        raise ValueError('scan inputs must share a device')
    return _ParallelAffine.apply(a, b, initial)


@lru_cache(maxsize=1)
def _compiled_parallel():
    # Opt-in only. fullgraph rejects graph breaks instead of silently reporting
    # a partly eager model as a fused implementation. Warmup is benchmarked apart.
    return torch.compile(parallel_affine, fullgraph=True, dynamic=False)


SCAN_BACKENDS = ('reference', 'parallel', 'compiled', 'fused')


def _fused_usable(state: torch.Tensor, *tensors: torch.Tensor) -> bool:
    if state.dtype != torch.float32:
        return False
    from .kernels import scan_triton
    return scan_triton.usable(state, *tensors)


def scan_values(a, b, initial, backend='reference'):
    if backend == 'reference':
        return _affine_values(a, b, initial)
    if backend == 'parallel':
        return parallel_affine(a, b, initial)
    if backend == 'compiled':
        return _compiled_parallel()(a, b, initial)
    if backend == 'fused':
        if _fused_usable(initial, a, b) and a.dtype == b.dtype == initial.dtype:
            from .kernels.scan_triton import affine_scan
            return affine_scan(a, b, initial)
        return parallel_affine(a, b, initial)
    raise ValueError('scan backend must be reference, parallel, compiled or fused')


def mingru_values(logits: torch.Tensor, state: torch.Tensor, backend: str = 'reference'):
    """States of the ScanLayer recurrence from gate logits [B,T,2D] = [gate|proposal].

    gate/proposal activations are stored at the logits' precision (as eager
    autocast does) before the fp32 state arithmetic. 'fused' runs the gate math
    and the scan in one Triton kernel per direction.
    """
    if backend == 'fused' and _fused_usable(state, logits):
        from .kernels.scan_triton import mingru_scan
        return mingru_scan(logits, state)
    gate, proposal = logits.chunk(2, dim=-1)
    # Keep recurrent arithmetic in the state precision; matmuls may use AMP.
    gate = torch.sigmoid(gate).to(state.dtype)
    proposal = torch.tanh(proposal).to(state.dtype)
    return scan_values(1.0 - gate, gate * proposal, state,
                       'parallel' if backend == 'fused' else backend)


def set_scan_backend(module: nn.Module, backend: str) -> int:
    """Switch every ScanFast/ScanLayer (and Gated DeltaNet core stack/layer,
    efficiency/gated_deltanet.py: 'fused' = the fla kernels) under `module`
    (execution only).

    Weights, state layout and the recurrence are unchanged; only the kernel
    (and float summation order) differs. Returns the number of layers switched.
    """
    if backend not in SCAN_BACKENDS:
        raise ValueError(f'scan backend must be one of {SCAN_BACKENDS}')
    from .gated_deltanet import GatedDeltaNetLayer, GatedDeltaNetStack
    count = 0
    for sub_module in module.modules():
        if isinstance(sub_module, (ScanLayer, GatedDeltaNetLayer)):
            sub_module.backend = backend
            count += 1
        elif isinstance(sub_module, (ScanFast, GatedDeltaNetStack)):
            sub_module.backend = backend
    return count


NORMS = ('layer', 'rms')


def make_norm(width: int, kind: str = 'layer', eps: float = 1e-5) -> nn.Module:
    """Pre-norm of a core block: 'layer' = nn.LayerNorm (the default; weight and
    bias), 'rms' = nn.RMSNorm (weight only, epsilon eps), the Llama/Qwen block norm,
    so a transformer's normalized MLP/attention inputs can be reproduced exactly
    (efficiency/transplant.py)."""
    if kind == 'layer':
        return nn.LayerNorm(width)
    if kind == 'rms':
        return nn.RMSNorm(width, eps=eps)
    raise ValueError(f'core norm must be one of {NORMS}')


class SwiGLU(nn.Module):
    """Pre-LN SwiGLU residual branch; returns only the branch (caller adds it).

    With zero_output=True the down projection starts at exactly zero, so adding
    the branch to an existing residual stream is function-preserving at init.
    norm='rms' uses an RMSNorm pre-norm (see make_norm).
    """
    def __init__(self, width: int, hidden: int, *, zero_output: bool = True, norm: str = 'layer',
                 norm_eps: float = 1e-5):
        super().__init__()
        if width < 1 or hidden < 1:
            raise ValueError('SwiGLU width and hidden must be positive')
        self.norm = make_norm(width, norm, norm_eps)
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        if zero_output:
            with torch.no_grad():
                self.down.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(self.norm(x)).chunk(2, dim=-1)
        return self.down(torch.nn.functional.silu(gate) * up)


def mlp_hidden_width(width: int, ratio: float) -> int:
    """Hidden width of an optional per-layer MLP; 0 disables it."""
    if ratio <= 0:
        return 0
    return max(1, int(round(float(ratio) * width)))


class ScanLayer(nn.Module):
    def __init__(self, width: int, residual_scale: float, backend: str = 'reference',
                 mlp_hidden: int = 0, norm: str = 'layer'):
        super().__init__()
        self.norm = make_norm(width, norm)
        self.gates = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width, bias=False)
        self.residual_scale = residual_scale
        self.backend = backend
        with torch.no_grad():
            self.gates.bias[:width].fill_(-1.0)
        # Optional channel mixing after the diagonal recurrence. Created only
        # when requested so default state_dict keys are unchanged; the output
        # projection is zero-initialized (function-preserving when grafted).
        self.mlp = SwiGLU(width, mlp_hidden, norm=norm) if mlp_hidden > 0 else None

    def forward(self, x: torch.Tensor, state: torch.Tensor, *, serial: bool = False,
                return_states: bool = False, return_output: bool = True):
        """return_output=False: the caller reads only the recurrent states; the
        residual output (out projection, MLP) is skipped and returned as None."""
        logits = self.gates(self.norm(x))
        if serial:
            gate, proposal = logits.chunk(2, dim=-1)
            # Keep recurrent arithmetic in the state precision; matmuls may use AMP.
            gate = torch.sigmoid(gate).to(state.dtype)
            proposal = torch.tanh(proposal).to(state.dtype)
            a, b = 1.0 - gate, gate * proposal
            rows, h = [], state
            for t in range(x.shape[1]):
                h = a[:, t] * h + b[:, t]
                rows.append(h)
            hs = torch.stack(rows, dim=1)
        else:
            hs = mingru_values(logits, state, self.backend)
        if not return_output:
            return (None, hs[:, -1], hs) if return_states else (None, hs[:, -1])
        output = x + self.residual_scale * self.out(hs).to(x.dtype)
        mlp = getattr(self, 'mlp', None)  # absent on modules pickled pre-option
        if mlp is not None:
            output = output + self.residual_scale * mlp(output).to(x.dtype)
        return (output, hs[:, -1], hs) if return_states else (output, hs[:, -1])


class ScanFast(nn.Module):
    """GRU-shaped public interface: (outputs, hidden[L,B,D])."""
    # Execution only (never a parameter or buffer; state_dict keys unchanged).
    # The hierarchical model sets it under segment_backend='compiled': then
    # gradient-enabled calls run the whole layer stack as one torch.compile'd
    # function (the fused Triton scan is an opaque custom op inside it). Same
    # math; only floating-point fusion/summation order differs.
    compile_training = False

    def __init__(self, width: int, layers: int, backend: str = 'reference', checkpoint_layers: bool = False,
                 mlp_ratio: float = 0.0, mlp_last_layer: bool = True, norm: str = 'layer'):
        super().__init__()
        if width < 1 or layers < 1:
            raise ValueError('width and layers must be positive')
        if backend not in SCAN_BACKENDS:
            raise ValueError('invalid scan backend')
        self.backend = backend
        if type(checkpoint_layers) is not bool:
            raise ValueError('checkpoint_layers must be boolean')
        self.checkpoint_layers = checkpoint_layers
        self.width, self.num_layers = width, layers
        if isinstance(mlp_ratio, bool) or not isinstance(mlp_ratio, (int, float))                 or not math.isfinite(mlp_ratio) or mlp_ratio < 0:
            raise ValueError('mlp_ratio must be a finite nonnegative number')
        hidden = mlp_hidden_width(width, mlp_ratio)
        # mlp_last_layer=False skips the top block's MLP for callers that read
        # the top recurrent state sequence instead of the residual output (the
        # top residual branch is then structurally unused).
        self.blocks = nn.ModuleList(
            ScanLayer(width, 1.0 / math.sqrt(layers), backend,
                      hidden if (mlp_last_layer or i < layers - 1) else 0, norm)
            for i in range(layers))

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None,
                *, serial: bool = False, depth: int | None = None,
                return_state_sequence: bool = False, return_output: bool = True):
        """return_output=False (callers that read only the recurrent states, e.g.
        the hierarchy's medium/core streams): the top layer's residual output is
        skipped and the first result is None."""
        if (self.compile_training and not serial and torch.is_grad_enabled()
                and not torch.compiler.is_compiling()):
            # hierarchical_patches owns the compiled-piece cache and settings.
            from .hierarchical_patches import _compiled
            return _compiled(_scan_fast_stack)(
                self, x, state, depth, return_state_sequence, False, return_output
            )
        return _scan_fast_stack(self, x, state, depth, return_state_sequence, serial,
                                return_output)


def _scan_fast_stack(module: ScanFast, x: torch.Tensor, state: torch.Tensor | None,
                     depth: int | None, return_state_sequence: bool, serial: bool = False,
                     return_output: bool = True):
    """ScanFast.forward's layer stack (the piece compiled under compile_training)."""
    depth = module.num_layers if depth is None else depth
    if not 1 <= depth <= module.num_layers:
        raise ValueError('depth outside recurrent stack')
    if x.ndim != 3 or x.shape[1] < 1 or x.shape[-1] != module.width:
        raise ValueError('expected nonempty [B,T,width]')
    if state is None:
        state = x.new_zeros(module.num_layers, x.shape[0], module.width)
    if state.shape != (module.num_layers, x.shape[0], module.width):
        raise ValueError('wrong recurrent state shape')
    next_states = []
    top_states = None
    for i, block in enumerate(module.blocks[:depth]):
        need_states = return_state_sequence and i == depth - 1
        output = return_output or i < depth - 1
        if module.checkpoint_layers and module.training and torch.is_grad_enabled():
            result = checkpoint(block, x, state[i], use_reentrant=False,
                                serial=serial, return_states=need_states, return_output=output)
        else:
            result = block(x, state[i], serial=serial, return_states=need_states,
                           return_output=output)
        x, h = result[:2]
        if need_states:
            top_states = result[2]
        next_states.append(h)
    next_states.extend(state[i] for i in range(depth, module.num_layers))
    hidden = torch.stack(next_states)
    return (x, hidden, top_states) if return_state_sequence else (x, hidden)
