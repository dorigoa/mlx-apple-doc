#!/usr/bin/env python3
"""
Patch idempotente per mlx-lm 0.31.3 (GenerationBatch._step): ogni N step di
decode materializza anche lo stato della prompt cache, così le catene lazy non
trattengono buffer Metal senza limite (rif. ml-explore/mlx-lm#1662).
N = variabile d'ambiente MLX_LM_CACHE_EVAL_INTERVAL (default 64).
"""
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile

EXPECTED_MD5 = "6089bff2a9f5b0038df0a907b024c34c"
MARKER = "# [local-patch] periodic cache-state eval"

OLD = "        mx.async_eval(self._next_tokens, self._next_logprobs, token_context)\n"
NEW = (
    f"        {MARKER} (ml-explore/mlx-lm#1662)\n"
    "        if not hasattr(self, \"_cache_eval_interval\"):\n"
    "            import os as _os\n"
    "            try:\n"
    "                self._cache_eval_interval = max(\n"
    "                    1, int(_os.environ.get(\"MLX_LM_CACHE_EVAL_INTERVAL\", \"64\"))\n"
    "                )\n"
    "            except ValueError:\n"
    "                self._cache_eval_interval = 64\n"
    "            self._cache_eval_step = 0\n"
    "        self._cache_eval_step += 1\n"
    "        _cache_state = []\n"
    "        if self._cache_eval_step % self._cache_eval_interval == 0:\n"
    "            _cache_state = [c.state for c in self.prompt_cache]\n"
    "        mx.async_eval(\n"
    "            self._next_tokens, self._next_logprobs, token_context, _cache_state\n"
    "        )\n"
)


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def main() -> int:
    spec = importlib.util.find_spec("mlx_lm")  # non esegue il package
    if spec is None or not spec.submodule_search_locations:
        print("ERRORE: mlx_lm non trovato in questo interprete", file=sys.stderr)
        return 1
    path = os.path.join(list(spec.submodule_search_locations)[0], "generate.py")

    with open(path, "rb") as f:
        raw = f.read()
    src = raw.decode("utf-8")

    if MARKER in src:
        print(f"GIÀ PATCHATO: {path} (md5 {md5(raw)})")
        return 0
    if md5(raw) != EXPECTED_MD5:
        print(f"ERRORE: md5 inatteso {md5(raw)} (atteso {EXPECTED_MD5}), non modifico",
              file=sys.stderr)
        return 2
    n = src.count(OLD)
    if n != 1:
        print(f"ERRORE: riga da sostituire trovata {n} volte (attesa 1)", file=sys.stderr)
        return 3

    new_src = src.replace(OLD, NEW)
    try:
        compile(new_src, path, "exec")
    except SyntaxError as e:
        print(f"ERRORE: il file patchato non compila: {e}", file=sys.stderr)
        return 4

    backup = path + ".orig-0.31.3"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path),
                               prefix=".generate.py.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new_src)
        shutil.copymode(path, tmp)
        os.replace(tmp, path)  # sostituzione atomica
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    with open(path, "rb") as f:
        print(f"OK: {path}\n    backup: {backup}\n    md5 nuovo: {md5(f.read())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
