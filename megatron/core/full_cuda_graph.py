# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Full iteration CUDA graph for training."""

import gc
import logging
import os
import time

import torch

from megatron.core.tensor_parallel.random import get_all_rng_states

logger = logging.getLogger(__name__)

# MLPERF_STEP_BOUNDARY=1 (overlay_stepb): cheaper step boundary for the full-iteration graph.
#   MLPERF_STEP_BOUNDARY_STAGE=stack (default with the switch) | dedup | off
#     dedup: stage each microbatch once per step instead of once per VPP chunk (every chunk of
#            microbatch b shares static buffer b and the last chunk's copy wins, so only that
#            one is staged), all copies under one stream switch and one wait_stream pair.
#     stack: dedup + the per-key static buffers of all microbatches are views into one
#            contiguous [num_microbatches, ...] tensor, filled by a single H2D copy per key
#            from a reused pinned staging tensor (torch.stack of the dataloader tensors).
#     The static buffers hold the same bytes as with the original per-microbatch copies.
#   MLPERF_STEP_BOUNDARY_EARLY_WAIT=1 (default with the switch): after a graph replay the
#     staging stream waits on an event recorded right after the replay (the last reader of the
#     static buffers) instead of on everything queued on the current stream since (e.g. eager
#     param casts at the step start), so the H2D copies overlap that work.
#   MLPERF_STEP_BOUNDARY_TIMING=1: print host time of the staging per step (rank 0 and last).
_SB = os.environ.get('MLPERF_STEP_BOUNDARY', '0') == '1'
_SB_STAGE = os.environ.get('MLPERF_STEP_BOUNDARY_STAGE', 'stack') if _SB else 'off'
_SB_EARLY_WAIT = _SB and os.environ.get('MLPERF_STEP_BOUNDARY_EARLY_WAIT', '1') == '1'
_SB_TIMING = os.environ.get('MLPERF_STEP_BOUNDARY_TIMING', '0') == '1'

# Process-wide handle so full-iter and optimizer graph captures share one pool and one
# non-default stream (per-stream alloc segments can inflate memory_reserved; see
# tools/debug_cuda_graph_pool_memory*.py).
_shared_graph_pool = None
_shared_capture_stream = None


def get_shared_capture_stream():
    """Return one `torch.cuda.Stream` for all full-iter and optimizer graph captures.

    Call after the target CUDA device is selected.
    """
    global _shared_capture_stream
    if _shared_capture_stream is None:
        _shared_capture_stream = torch.cuda.Stream()
    return _shared_capture_stream


def get_shared_graph_pool():
    """Return a process-wide handle so all call sites share one graph memory pool.

    `torch.cuda.graph_pool_handle()` returns a new pool each time; this lazy singleton
    ensures e.g. full-iteration and optimizer captures reuse the same pool.
    """
    global _shared_graph_pool
    if _shared_graph_pool is None:
        _shared_graph_pool = torch.cuda.graph_pool_handle()
    return _shared_graph_pool


def get_graph_pool(use_single_mempool):
    """Return graph pool handle for full-iter/optimizer graph capture.

    When `use_single_mempool` is True, train/eval and optimizer captures reuse one
    process-wide pool. Otherwise, each capture call gets a new pool handle.
    """
    if use_single_mempool:
        return get_shared_graph_pool()
    return torch.cuda.graph_pool_handle()


# The below functions traverse through nested data structures (tuples, lists, dicts)
# present in src and creates a deep copy where all PyTorch tensors are cloned,
# detached from the computation graph, and moved to CUDA device. Non-tensor objects
# are returned as-is.


def copy_tensors_in_struct(src):
    """Copy src to new tensors."""
    if isinstance(src, tuple):
        return tuple(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, list):
        return list(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, dict):
        return {k: copy_tensors_in_struct(src[k]) for k in src}
    elif isinstance(src, torch.Tensor):
        return src.clone().detach().cuda()
    else:
        return src


def clone_tensors_in_struct(tgt, src):
    """Copy src to pre-existing tensors in tgt."""
    if isinstance(src, tuple):
        raise Exception(f"Unsupported copy for tuple yet: {type(src)}")
    elif isinstance(src, list):
        for i in range(len(src)):
            if isinstance(src[i], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[i], src[i])
            else:
                tgt[i] = src[i]
    elif isinstance(src, dict):
        for k in src:
            if isinstance(src[k], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[k], src[k])
            else:
                tgt[k] = src[k]
    elif isinstance(src, torch.Tensor):
        tgt.copy_(src, non_blocking=True)
    else:
        raise Exception(f"Expect top-level as container type but got: {type(src)}")


# Class to copy dataloader output to static CUDA tensors for CUDA graph input. This
# maintains separate static buffers for training and validation CUDA graphs.
class StaticBufferLoader:
    """Load data to static buffers."""

    static_buffers: dict = {'training': [], 'validation': []}

    # overlay_stepb state, class-level like static_buffers (the warmup callback's wrapper
    # captures the graph and fills the buffers; the training loop's wrapper replays it).
    sb_big: dict = {}  # stage -> {key: cuda tensor [num_mb, ...]}; static buffers are its views
    sb_pin: dict = {}  # stage -> {key: pinned cpu tensor [num_mb, ...]} for cpu sources
    sb_h2d_done: dict = {}  # stage -> event after the last H2D out of the pinned tensors
    consumed_event: dict = {}  # stage -> event after the last graph replay (reader of the buffers)

    def __init__(self):
        self.stream = torch.cuda.Stream()

    @staticmethod
    def _normalize(inputs):
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]
        assert isinstance(inputs, dict)
        return inputs

    sb_views_ok: dict = {}  # stage -> (num_mb, frozenset of keys verified to be views of sb_big)

    @staticmethod
    def _any_cuda(x):
        if isinstance(x, torch.Tensor):
            return x.is_cuda
        if isinstance(x, (tuple, list)):
            return any(StaticBufferLoader._any_cuda(i) for i in x)
        if isinstance(x, dict):
            return any(StaticBufferLoader._any_cuda(i) for i in x.values())
        return False

    def _wait_for_consumers(self, stage, all_cpu):
        # The early wait only orders after the readers of the static buffers, so it needs all
        # sources to be host tensors (nothing produced on the current stream to wait for).
        ev = StaticBufferLoader.consumed_event.pop(stage, None)
        if ev is not None and _SB_EARLY_WAIT and all_cpu:
            self.stream.wait_event(ev)
        else:
            self.stream.wait_stream(torch.cuda.current_stream())

    @staticmethod
    def _stackable(norm):
        """{key: (shape, dtype)} of keys that hold, in every microbatch, a contiguous cpu/cuda
        tensor of one shape, dtype and device."""
        keys = {}
        for k, v in norm[0].items():
            if not isinstance(v, torch.Tensor) or not v.is_contiguous():
                continue
            if v.device.type not in ('cpu', 'cuda'):
                continue
            ok = True
            for d in norm[1:]:
                w = d.get(k, None)
                if (
                    not isinstance(w, torch.Tensor)
                    or w.shape != v.shape
                    or w.dtype != v.dtype
                    or w.device != v.device
                    or not w.is_contiguous()
                ):
                    ok = False
                    break
            if ok:
                keys[k] = (tuple(v.shape), v.dtype)
        return keys

    @staticmethod
    def _stage_rest(static, inputs, skip):
        """Original per-microbatch semantics (StaticBufferLoader.__call__ else-branch) for the
        keys not in skip; runs on the current (staging) stream."""
        for k in inputs.keys():
            if k in skip:
                continue
            if k not in static:
                if isinstance(inputs[k], torch.Tensor):
                    static[k] = torch.empty_like(inputs[k], device="cuda")
                else:
                    static[k] = inputs[k]
        clone_tensors_in_struct(static, {k: v for k, v in inputs.items() if k not in skip})

    def load_all(self, batches, stage):
        """Stage one input dict per microbatch (list index = microbatch) into the static buffers.

        Leaves the same bytes in the static buffers as calling self(batches[b], stage, b) for
        every b in order, with one wait_stream pair and (stack mode) one H2D copy per key.
        Returns the list of static buffer dicts.
        """
        assert stage in ['training', 'validation']
        bufs = StaticBufferLoader.static_buffers[stage]
        norm = [self._normalize(x) for x in batches]
        n = len(norm)
        if n == 0:
            return []
        if len(bufs) == 0 and _SB_STAGE == 'stack':
            # First call: allocate one contiguous [n, ...] buffer per stackable key; the
            # per-microbatch static buffers are its row views (contiguous, same shape/dtype
            # as the clone the original path makes).
            stk = self._stackable(norm)
            big = {k: torch.empty((n,) + shp, dtype=dt, device='cuda') for k, (shp, dt) in stk.items()}
            for b in range(n):
                bufs.append(
                    {k: (big[k][b] if k in big else copy_tensors_in_struct(v)) for k, v in norm[b].items()}
                )
            StaticBufferLoader.sb_big[stage] = big
            StaticBufferLoader.sb_pin[stage] = {}
        if len(bufs) < n:
            # more microbatches than buffers (not expected): original path
            return [self(norm[b], stage, b) for b in range(n)]
        big = StaticBufferLoader.sb_big.get(stage, {}) if _SB_STAGE == 'stack' else {}
        stk = {}
        if big:
            cand = self._stackable(norm)
            ok = StaticBufferLoader.sb_views_ok.get(stage)
            if ok is None or ok[0] != n:
                ok = (
                    n,
                    frozenset(
                        k
                        for k, t in big.items()
                        if t.shape[0] == n
                        and all(
                            isinstance(bufs[b].get(k), torch.Tensor)
                            and bufs[b][k].data_ptr() == t[b].data_ptr()
                            and bufs[b][k].shape == t[b].shape
                            for b in range(n)
                        )
                    ),
                )
                StaticBufferLoader.sb_views_ok[stage] = ok
            for k, (shp, dt) in cand.items():
                t = big.get(k)
                if k in ok[1] and tuple(t.shape[1:]) == shp and t.dtype == dt:
                    stk[k] = norm[0][k].device.type
        pin = StaticBufferLoader.sb_pin.setdefault(stage, {})
        if any(dev == 'cpu' for dev in stk.values()):
            done = StaticBufferLoader.sb_h2d_done.get(stage)
            if done is not None:
                done.synchronize()  # pinned staging tensors are free again (long done)
            for k, dev in stk.items():
                if dev != 'cpu':
                    continue
                if k not in pin:
                    pin[k] = torch.empty(big[k].shape, dtype=big[k].dtype, pin_memory=True)
                torch.stack([d[k] for d in norm], out=pin[k])
        self._wait_for_consumers(stage, not any(self._any_cuda(d) for d in norm))
        with torch.cuda.stream(self.stream):
            for k, dev in stk.items():
                if dev == 'cpu':
                    big[k].copy_(pin[k], non_blocking=True)
                else:
                    torch.stack([d[k] for d in norm], out=big[k])
            skip = set(stk)
            for b in range(n):
                self._stage_rest(bufs[b], norm[b], skip)
            if stk:
                ev = torch.cuda.Event()
                ev.record(self.stream)
                StaticBufferLoader.sb_h2d_done[stage] = ev
        torch.cuda.current_stream().wait_stream(self.stream)
        return bufs[:n]

    def __call__(self, inputs, stage, microbatch):
        assert stage in ['training', 'validation']
        assert microbatch <= len(StaticBufferLoader.static_buffers[stage])
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]

        assert isinstance(inputs, dict)
        if microbatch == len(StaticBufferLoader.static_buffers[stage]):
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                StaticBufferLoader.static_buffers[stage].append(copy_tensors_in_struct(inputs))
        else:

            for k in inputs.keys():
                if k not in StaticBufferLoader.static_buffers[stage][microbatch]:
                    if isinstance(inputs[k], torch.Tensor):
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = torch.empty_like(
                            inputs[k], device="cuda"
                        )
                    else:
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = inputs[k]

            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                clone_tensors_in_struct(
                    StaticBufferLoader.static_buffers[stage][microbatch], inputs
                )
        torch.cuda.current_stream().wait_stream(self.stream)
        return StaticBufferLoader.static_buffers[stage][microbatch]


class FullCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    curr_iteration = {'training': 0, 'validation': 0}
    cuda_graph = {'training': None, 'validation': None}
    result = {'training': None, 'validation': None}

    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=1, use_single_mempool=False):
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.use_single_mempool = use_single_mempool

    def _sb_data_read(self, data_iterator, model, training, num_microbatches):
        """overlay_stepb data_read: same iterator consumption and final buffer contents."""
        stage = 'training' if training else 'validation'
        t0 = time.perf_counter() if _SB_TIMING else None
        single = not isinstance(model, list) or len(model) == 1
        if single:
            assert not isinstance(data_iterator, list) or len(data_iterator) == 1
            its = [data_iterator if not isinstance(data_iterator, list) else data_iterator[0]]
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            its = data_iterator
        per_chunk = []
        for it in its:
            per_chunk.append(None if it is None else [next(it) for _ in range(num_microbatches)])
        # Every chunk stages microbatch b into the same static buffer b, in chunk order, so the
        # last chunk with data determines the final contents.
        finals = None
        for bl in per_chunk:
            if bl is not None:
                finals = bl
        statics = None if finals is None else self.static_loader.load_all(finals, stage)
        if single:
            out = [None] if statics is None else [iter(statics)]
        else:
            out = [None if bl is None else iter(list(statics)) for bl in per_chunk]
        if _SB_TIMING:
            self._sb_t_stage = time.perf_counter() - t0
        return out

    def data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs from Dataloader and copy to static buffers."""
        if _SB_STAGE != 'off':
            return self._sb_data_read(data_iterator, model, training, num_microbatches)
        t0 = time.perf_counter() if _SB_TIMING else None
        out = self._orig_data_read(data_iterator, model, training, num_microbatches)
        if _SB_TIMING:
            self._sb_t_stage = time.perf_counter() - t0
        return out

    def _orig_data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs from Dataloader and copy to static buffers."""
        if not isinstance(model, list) or len(model) == 1:
            assert not isinstance(data_iterator, list) or len(data_iterator) == 1
            iterator0 = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
            data_list = []
            if iterator0 is not None:
                for b in range(num_microbatches):
                    data_list.append(
                        self.static_loader(
                            next(iterator0), 'training' if training else 'validation', b
                        )
                    )
                data_list = [iter(data_list)]
            else:
                data_list.append(None)
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            data_list = []
            for i in range(len(model)):
                if data_iterator[i] is not None:
                    data_list_i = []
                    for b in range(num_microbatches):
                        data_list_i.append(
                            self.static_loader(
                                next(data_iterator[i]), 'training' if training else 'validation', b
                            )
                        )
                    data_list.append(iter(data_list_i))
                else:
                    data_list.append(None)
        return data_list

    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'forward_backward_func does not accept positional args'
        assert all(
            [
                kwarg in kwargs
                for kwarg in [
                    'model',
                    'data_iterator',
                    'num_microbatches',
                    'seq_length',
                    'forward_only',
                ]
            ]
        )
        model = kwargs['model']
        num_microbatches = kwargs['num_microbatches']

        training = not kwargs['forward_only']
        data_iterator = kwargs['data_iterator']
        data_list = self.data_read(data_iterator, model, training, num_microbatches)
        kwargs['data_iterator'] = data_list

        training_str = 'training' if training else 'validation'
        curr_iteration = self.curr_iter(training_str)
        if curr_iteration == self.cuda_graph_warmup_steps:
            logger.info(f'Capture CUDA graph for {training_str}!!!')
            if hasattr(torch.autograd.graph, 'set_override_stale_capture_stream'):
                torch.autograd.graph.set_override_stale_capture_stream(True)
            else:
                logger.warning(
                    'torch.autograd.graph.set_override_stale_capture_stream is not '
                    'available in this PyTorch version; CUDA graph capture may fail '
                    'if autograd nodes hold stale references to non-capturing streams. '
                    'Upgrade to a PyTorch build that includes pytorch/pytorch#180090.'
                )
            torch.distributed.barrier()
            assert FullCudaGraphWrapper.cuda_graph[training_str] is None
            FullCudaGraphWrapper.cuda_graph[training_str] = torch.cuda.CUDAGraph()
            for _, state in get_all_rng_states().items():
                FullCudaGraphWrapper.cuda_graph[training_str].register_generator_state(state)
            torch.cuda.synchronize()
            capture_stream = get_shared_capture_stream()
            with torch.cuda.graph(
                FullCudaGraphWrapper.cuda_graph[training_str],
                stream=capture_stream,
                pool=get_graph_pool(self.use_single_mempool),
                capture_error_mode="thread_local",
            ):
                FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(
                    *args, **kwargs
                )
            torch.cuda.synchronize()
            torch.distributed.barrier()
            logger.info(f'CUDA graph capture done for {training_str}!!!')
        if FullCudaGraphWrapper.cuda_graph[training_str] is None:
            FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(*args, **kwargs)
        else:
            FullCudaGraphWrapper.cuda_graph[training_str].replay()
            if _SB_EARLY_WAIT:
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream())
                StaticBufferLoader.consumed_event[training_str] = ev
        if _SB_TIMING:
            _r = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            _w = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            if _r in (0, _w - 1):
                print(
                    f"[stepb] rank {_r} {training_str} iter {self.curr_iter(training_str)} "
                    f"stage_host_ms {1e3 * getattr(self, '_sb_t_stage', 0.0):.2f} mode {_SB_STAGE}",
                    flush=True,
                )
        self.next_iter(training_str)
        return FullCudaGraphWrapper.result[training_str]

    def curr_iter(self, stage):
        """Return current training/validation iteration."""
        return FullCudaGraphWrapper.curr_iteration[stage]

    def next_iter(self, stage):
        """Increment current training/validation iteration."""
        FullCudaGraphWrapper.curr_iteration[stage] += 1

    def reset_cuda_graph(self, stage=None):
        """Reset CUDA graph."""
        if stage is None or stage == 'training':
            if FullCudaGraphWrapper.cuda_graph['training'] is not None:
                del FullCudaGraphWrapper.cuda_graph['training']
                FullCudaGraphWrapper.cuda_graph['training'] = None
            FullCudaGraphWrapper.result['training'] = None
            FullCudaGraphWrapper.curr_iteration['training'] = 0
        if stage is None or stage == 'validation':
            if FullCudaGraphWrapper.cuda_graph['validation'] is not None:
                del FullCudaGraphWrapper.cuda_graph['validation']
                FullCudaGraphWrapper.cuda_graph['validation'] = None
            FullCudaGraphWrapper.result['validation'] = None
            FullCudaGraphWrapper.curr_iteration['validation'] = 0
        gc.collect()
