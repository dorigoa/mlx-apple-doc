#!/usr/bin/env python3
"""

execute on both mac nodes: /Volumes/Home/Alvise/miniforge3/envs/mlx/bin/python mlx_leak_shim.py --self-test

expected output:

--- fase unpatched ---
    CRASH allo step 10395: [metal::malloc] Resource limit (499000) exceeded.
--- fase patched ---
    OK: 12000 step x 48 layer, valori corretti
    [leak-shim] rank=? patch applicata (mlx-lm 0.31.3)
PASS: leak riprodotto senza patch, assente con la patch

----------------

mlx_leak_shim.py - avvia mlx_lm.server con una patch al leak di buffer Metal
dei modelli ibridi (ArraysCache: qwen3_5 / Qwen3.6 / Qwen3.8, qwen3_next, ...).

Causa (mlx-lm 0.31.3): ArraysCache.advance() esegue `-= N` lazy su
left_padding/lengths a ogni step; per i layer il cui metadata non viene mai
letto la catena non viene mai valutata e ogni nodo trattiene un buffer Metal
-> "[metal::malloc] Resource limit (499000) exceeded" dopo ~10k token.

Fix: dopo ogni BatchGenerator.next() si valutano (async) solo left_padding e
lengths delle cache avanzate nello step. La catena collassa; i valori non cambiano.

Uso:
  python mlx_leak_shim.py [argomenti di mlx_lm.server]
  python mlx_leak_shim.py --self-test
Variabili d'ambiente:
  MLX_LEAK_SHIM=0         non applica la patch (per confronti A/B)
  MLX_LEAK_SHIM_STRICT=1  esce con errore se il codice installato non e' quello atteso
"""

import functools
import inspect
import os
import subprocess
import sys
import threading

TAG = "[leak-shim]"
EXPECTED = ("self.lengths -= N", "self.left_padding -= N")

_pending = {}
_lock = threading.Lock()


def _log(msg):
    print(f"{TAG} rank={os.environ.get('MLX_RANK', '?')} {msg}",
          file=sys.stderr, flush=True)


def _flush_pending():
    import mlx.core as mx
    with _lock:
        caches = list(_pending.values())
        _pending.clear()
    arrays = [a for c in caches
              for a in (getattr(c, "left_padding", None), getattr(c, "lengths", None))
              if isinstance(a, mx.array)]
    if arrays:
        mx.async_eval(arrays)


def apply_patch():
    if os.environ.get("MLX_LEAK_SHIM", "1") == "0":
        _log("MLX_LEAK_SHIM=0: patch NON applicata")
        return False

    import mlx_lm
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models.cache import ArraysCache

    if getattr(ArraysCache.advance, "_leak_shim", False):
        return True

    version = getattr(mlx_lm, "__version__", "?")
    try:
        src = inspect.getsource(ArraysCache.advance)
    except (OSError, TypeError):
        src = ""
    if not all(p in src for p in EXPECTED) or not callable(getattr(BatchGenerator, "next", None)):
        msg = f"mlx-lm {version}: codice diverso da quello atteso, patch NON applicata"
        if os.environ.get("MLX_LEAK_SHIM_STRICT") == "1":
            _log(msg + " (STRICT: esco)")
            sys.exit(2)
        _log(msg)
        return False

    orig_advance = ArraysCache.advance

    @functools.wraps(orig_advance)
    def advance(self, N):
        orig_advance(self, N)
        with _lock:
            _pending[id(self)] = self

    orig_next = BatchGenerator.next

    @functools.wraps(orig_next)
    def next_(self, *args, **kwargs):
        out = orig_next(self, *args, **kwargs)
        _flush_pending()
        return out

    advance._leak_shim = True
    next_._leak_shim = True
    ArraysCache.advance = advance
    BatchGenerator.next = next_
    _log(f"patch applicata (mlx-lm {version})")
    return True


# ---------------- self-test (nessun modello, nessun jaccl) ----------------
N_LAYERS = 48      # layer lineari di Qwen3.8-27B (64 x 3/4)
N_STEPS = 12_000   # oltre 499000/48 ~ 10.400


def _phase(patched):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    if patched and not apply_patch():
        print("patch non applicabile")
        return 1

    caches = []
    for _ in range(N_LAYERS):
        c = ArraysCache(2)
        c.left_padding = mx.array([0], dtype=mx.int32)
        caches.append(c)

    step = 0
    try:
        for step in range(1, N_STEPS + 1):
            for c in caches:
                c.advance(1)
            if patched:
                _flush_pending()
    except RuntimeError as e:
        print(f"CRASH allo step {step}: {e}")
        return 3

    mx.eval([c.left_padding for c in caches])
    vals = {c.left_padding[0].item() for c in caches}
    if vals != {-N_STEPS}:
        print(f"VALORI ERRATI: {vals} (atteso {-N_STEPS})")
        return 5
    print(f"OK: {N_STEPS} step x {N_LAYERS} layer, valori corretti")
    return 0


def _self_test():
    results = {}
    for phase in ("unpatched", "patched"):
        print(f"--- fase {phase} ---", flush=True)
        try:
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--self-test-phase", phase],
                capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            print("    TIMEOUT")
            results[phase] = None
            continue
        for line in (p.stdout + p.stderr).strip().splitlines()[-5:]:
            print("   ", line)
        results[phase] = p.returncode

    if results.get("unpatched") == 3 and results.get("patched") == 0:
        print("PASS: leak riprodotto senza patch, assente con la patch")
        return 0
    if results.get("unpatched") == 0 and results.get("patched") == 0:
        print("INCONCLUSIVO: la simulazione non riproduce il leak su questa versione")
        return 4
    print(f"FAIL: codici di uscita {results}")
    return 1


def main():
    args = sys.argv[1:]
    if args[:1] == ["--self-test"]:
        return _self_test()
    if args[:1] == ["--self-test-phase"]:
        if len(args) < 2 or args[1] not in ("patched", "unpatched"):
            print("uso: --self-test-phase patched|unpatched", file=sys.stderr)
            return 2
        return _phase(args[1] == "patched")

    apply_patch()
    sys.argv = ["mlx_lm.server"] + args
    try:
        from mlx_lm.server import main as server_main
    except ImportError:
        import runpy
        runpy.run_module("mlx_lm.server", run_name="__main__", alter_sys=True)
        return 0
    server_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
