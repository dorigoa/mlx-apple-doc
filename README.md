# MLX distribuito con RDMA su Thunderbolt 5 — 2× Mac Studio

Sep 24, 2026 · @Alvise Dorigo

Configurazione verificata: 2× Mac Studio M4 Max 36 GB in Thunderbolt 5, mlx 0.32.2 + mlx-lm 0.31.3, backend JACCL (RDMA), tensor parallel su 2 nodi. Qwen3.6-27B-8bit genera 22,4 tok/s.

## 1. Prerequisiti e abilitazione RDMA

Servono macOS 26.2 o successivo su entrambi i Mac e un cavo Thunderbolt 5 diretto tra i due.

Abilitazione RDMA, una volta per Mac:

1. Avvio in recoveryOS: Mac spento, tieni premuto il tasto di accensione fino a "Opzioni".
2. Menu Utility → Terminale.
3. Esegui `rdma_ctl enable`.
4. Riavvia normalmente.

Verifica (su entrambi):

```bash
sw_vers -productVersion
rdma_ctl status
ibv_devinfo | grep -E 'hca_id|state'
```

Atteso: versione ≥ 26.2, `enabled`, device `rdma_en3` (Mac1) / `rdma_en4` (Mac2) in stato `PORT_ACTIVE`.

## 2. Rete Thunderbolt: IP, MTU 9000, bridge

Mac1 è il rank 0: da qui si lanciano tutti i job.

| Nodo | Interfaccia | IP | Device RDMA |
| --- | --- | --- | --- |
| Mac1 (rank 0) | en3 | 192.168.20.1/24 | rdma\_en3 |
| Mac2 (rank 1) | en4 | 192.168.20.2/24 | rdma\_en4 |

Disattiva il Bridge Thunderbolt (consigliato dalla documentazione MLX anche se RDMA non usa TCP/IP), su entrambi:

```bash
sudo networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off
```

IP e MTU (non persistenti; per renderli permanenti usa Impostazioni di Sistema → Rete sul servizio di en3/en4, mappatura con `networksetup -listallhardwareports`):

```bash
# Mac1
sudo ifconfig en3 inet 192.168.20.1 netmask 255.255.255.0 mtu 9000
# Mac2
sudo ifconfig en4 inet 192.168.20.2 netmask 255.255.255.0 mtu 9000
```

Verifica jumbo frame da Mac1 (8972 = 9000 − 28 byte di header IP+ICMP, `-D` vieta la frammentazione):

```bash
ping -c3 -D -s 7168 192.168.20.2
```

Alternativa automatica: `mlx.distributed_config --backend jaccl --hosts <h1>,<h2> --over thunderbolt --auto-setup --output hosts.json` configura rete e hostfile, ma richiede sudo senza password e usa l'IP di en0 per il rank 0.

## 3. Python: Miniforge e ambiente mlx

L'ambiente deve essere identico e nello stesso percorso sui due Mac: `mlx.launch` avvia via SSH lo stesso eseguibile Python su entrambi.

Installazione Miniforge (su entrambi, se assente):

```bash
curl -fsSLO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init "$(basename "$SHELL")"
exec "$SHELL" -l
```

Ambiente e pacchetti (su entrambi, versioni fissate per riproducibilità):

```bash
conda create -y -n mlx python=3.12
conda activate mlx
pip install mlx==0.32.2 mlx-lm==0.31.3
python -c "import mlx.core as mx, mlx_lm; print(mx.__version__, mlx_lm.__version__)"
which python
```

Atteso: `0.32.2 0.31.3` e percorso `$HOME/miniforge3/envs/mlx/bin/python` uguale sui due Mac (qui `/Volumes/Home/Alvise/...`).

## 4. Hugging Face CLI (hf): installazione, token, download

Il comando `hf` fa parte del pacchetto `huggingface_hub`, già installato come dipendenza di mlx-lm.

Aggiornamento e verifica (su entrambi, env `mlx` attivo):

```bash
pip install -U huggingface_hub
which hf
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

Token: huggingface.co → Settings → Access Tokens → New token, tipo **Read**. Esportalo come `HF_TOKEN` nella shell.

Login (su entrambi). Il token viene salvato in `~/.cache/huggingface/token` e resta valido anche nelle sessioni SSH non interattive, dove `HF_TOKEN` potrebbe non essere definito:

```bash
hf auth login --token "$HF_TOKEN"
hf auth whoami
```

Per i modelli ad accesso limitato (gated) accetta prima la licenza sulla pagina del modello.

Architettura di un modello senza scaricarlo (serve per la sezione 10):

```bash
repo=mlx-community/Qwen3-32B-8bit
curl -sSfL -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/$repo/resolve/main/config.json" | grep -m1 '"model_type"'
```

Senza token HF risponde "Invalid username or password" anche per repo inesistenti; con `-f` curl mostra il codice reale (401/404).

La cache dei modelli è in `~/.cache/huggingface/hub`; si sposta con la variabile `HF_HOME`.

## 5. SSH senza password

Attiva "Login remoto" su entrambi: Impostazioni di Sistema → Generali → Condivisione.

Su Mac1 (il launcher si collega via SSH anche a sé stesso):

```bash
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id 192.168.20.1
ssh-copy-id 192.168.20.2
ssh 192.168.20.2 'hostname; ~/miniforge3/envs/mlx/bin/python -c "import mlx.core as mx; print(mx.__version__)"'
```

Atteso: nessuna richiesta di password, hostname di Mac2 e `0.32.2`.

## 6. Hostfile e test JACCL

L'hostfile JACCL contiene, per ogni nodo: host SSH, IP del rank 0 (coordinatore TCP, raggiungibile da tutti) e device RDMA verso ogni altro nodo (`null` verso sé stesso).

Su Mac1:

```bash
mkdir -p ~/mlx-dist && cd ~/mlx-dist
cat > hosts.json <<'EOF'
[
  {"ssh": "192.168.20.1", "ips": ["192.168.20.1"], "rdma": [null, "rdma_en3"]},
  {"ssh": "192.168.20.2", "ips": [], "rdma": ["rdma_en4", null]}
]
EOF
cat > test_jaccl.py <<'EOF'
import mlx.core as mx
g = mx.distributed.init(strict=True, backend="jaccl")
x = mx.distributed.all_sum(mx.ones(4) * (g.rank() + 1))
mx.eval(x)
print(f"rank {g.rank()}/{g.size()}: {x.tolist()}")
EOF
ssh 192.168.20.2 'mkdir -p ~/mlx-dist' && scp test_jaccl.py 192.168.20.2:mlx-dist/
mlx.launch --verbose --backend jaccl --hostfile hosts.json --env MLX_METAL_FAST_SYNCH=1 -- \
  $HOME/miniforge3/envs/mlx/bin/python $HOME/mlx-dist/test_jaccl.py
```

Atteso (1 + 2 = 3 su ogni elemento):

```
rank 0/2: [3.0, 3.0, 3.0, 3.0]
rank 1/2: [3.0, 3.0, 3.0, 3.0]
```

Dopo `--` va sempre il percorso assoluto di Python: senza, il launcher esegue lo script come comando e fallisce con `Permission denied`. Lo script deve esistere nello stesso percorso su entrambi i nodi.

## 7. Script tp\_generate.py

Il pacchetto pip non include `mlx_lm.examples`: questo script lo sostituisce. Usa `sharded_load` (pesi divisi tra i nodi) e `stream_generate`.

Salvalo in `~/mlx-dist/tp_generate.py` su Mac1, poi copialo con `scp ~/mlx-dist/tp_generate.py 192.168.20.2:mlx-dist/`.

```python
import argparse, sys
import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.utils import sharded_load
from mlx_lm.sample_utils import make_sampler, make_logits_processors

p = argparse.ArgumentParser(description="Inferenza tensor-parallel MLX/JACCL")
p.add_argument("--model", required=True)
src = p.add_mutually_exclusive_group()
src.add_argument("--prompt", default="Ciao")
src.add_argument("--prompt-file")
p.add_argument("--system")
p.add_argument("--max-tokens", type=int, default=512)
p.add_argument("--no-think", action="store_true")
p.add_argument("--temp", type=float, default=0.0)
p.add_argument("--top-p", type=float, default=0.0)
p.add_argument("--top-k", type=int, default=0)
p.add_argument("--min-p", type=float, default=0.0)
p.add_argument("--rep-penalty", type=float)
p.add_argument("--rep-context", type=int, default=20)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max-kv-size", type=int)
p.add_argument("--kv-bits", type=int, choices=[4, 8])
p.add_argument("--kv-group-size", type=int, default=64)
p.add_argument("--quantized-kv-start", type=int, default=0)
p.add_argument("--prefill-step-size", type=int, default=2048)
a = p.parse_args()

g = mx.distributed.init(strict=True, backend="jaccl")
r0 = g.rank() == 0
mx.random.seed(a.seed)  # stesso seme su tutti i rank: campionamento identico

def die(msg):
    print(f"[rank {g.rank()}] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)

if a.max_tokens <= 0:
    die("--max-tokens deve essere > 0")

if a.prompt_file:
    try:
        with open(a.prompt_file, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        die(f"prompt-file: {e}")
else:
    text = a.prompt

try:
    model, tok = sharded_load(a.model, None, g)
except Exception as e:
    die(f"load failed: {e}")

msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": text}]
if tok.chat_template:
    kw = {"enable_thinking": False} if a.no_think else {}
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, **kw)
else:
    prompt = text

gen_kw = dict(
    max_tokens=a.max_tokens,
    sampler=make_sampler(temp=a.temp, top_p=a.top_p, min_p=a.min_p, top_k=a.top_k),
    logits_processors=make_logits_processors(repetition_penalty=a.rep_penalty,
                                             repetition_context_size=a.rep_context),
    max_kv_size=a.max_kv_size,
    prefill_step_size=a.prefill_step_size,
    kv_bits=a.kv_bits,
    kv_group_size=a.kv_group_size,
    quantized_kv_start=a.quantized_kv_start,
)

r = None
try:
    for r in stream_generate(model, tok, prompt, **gen_kw):
        if r0:
            print(r.text, end="", flush=True)
except Exception as e:
    die(f"generation failed: {e}")

if r0 and r:
    print(f"\n[world={g.size()}] prompt {r.prompt_tokens} tok @ {r.prompt_tps:.1f} tok/s | "
          f"gen {r.generation_tokens} tok @ {r.generation_tps:.1f} tok/s | peak {r.peak_memory:.2f} GB")
```

## 8. Wrapper tp.sh, download modello e lancio

Il wrapper evita di ricopiare ogni volta la lunga riga di `mlx.launch`, dove un errore di copia-incolla basta a far fallire il lancio. Salvalo in `~/mlx-dist/tp.sh` su Mac1:

```bash
#!/usr/bin/env bash
# Uso: ./tp.sh --model <repo> [opzioni di tp_generate.py]
set -euo pipefail
ENV="$HOME/miniforge3/envs/mlx"
cd "$HOME/mlx-dist"
[[ -f hosts.json ]] || { echo "hosts.json mancante" >&2; exit 1; }
[[ -f tp_generate.py ]] || { echo "tp_generate.py mancante" >&2; exit 1; }
exec "$ENV/bin/mlx.launch" --backend jaccl --hostfile hosts.json \
  --env MLX_METAL_FAST_SYNCH=1 --env HF_HUB_OFFLINE=1 -- \
  "$ENV/bin/python" "$HOME/mlx-dist/tp_generate.py" "$@"
```

```bash
chmod +x ~/mlx-dist/tp.sh
```

`MLX_METAL_FAST_SYNCH=1` velocizza la sincronizzazione tra GPU e CPU. `HF_HUB_OFFLINE=1` evita controlli di rete verso Hugging Face e richiede il modello già scaricato.

Download del modello su entrambi i nodi, in parallelo da Mac1:

```bash
export MLX_METAL_FAST_SYNCH=1
export HF_HUB_OFFLINE=1
M=mlx-community/Qwen3.6-27B-8bit
hf download "$M" --quiet &
ssh 192.168.20.2 "~/miniforge3/envs/mlx/bin/hf download $M --quiet" &
wait
d="models--${M//\//--}"
du -sh ~/.cache/huggingface/hub/$d
ssh 192.168.20.2 "du -sh ~/.cache/huggingface/hub/$d"
```

Le due dimensioni devono coincidere (29 GB per questo modello).

Lancio:

```bash
~/mlx-dist/tp.sh --model $M --prompt "Spiega in 5 frasi il teorema del viriale." --max-tokens 2048
```

Risultati misurati:

| Modello | Generazione (tok/s) | Picco per nodo (GB) |
| --- | --- | --- |
| mlx-community/Qwen3.6-27B-8bit | 22,4 | non rilevato |
| mlx-community/Llama-3.2-1B-Instruct-4bit | 497,8 | 0,56 |

## 9. Opzioni di tp\_generate.py

| Opzione | Effetto |
| --- | --- |
| `--model REPO` | Repo Hugging Face o percorso locale (obbligatorio) |
| `--prompt "…"` | Prompt inline |
| `--prompt-file F` | Prompt da file, presente nello stesso percorso su entrambi i nodi |
| `--system "…"` | Prompt di sistema |
| `--max-tokens N` | Token massimi generati (default 512) |
| `--no-think` | Disattiva il blocco di ragionamento, se il template lo supporta |
| `--temp T` | Temperatura; 0 = sempre il token più probabile (default) |
| `--top-p P` | Campiona tra i token che coprono la probabilità cumulata P (0 = off) |
| `--top-k K` | Campiona tra i K token più probabili (0 = off) |
| `--min-p M` | Scarta i token con probabilità < M × quella del migliore (0 = off) |
| `--rep-penalty R` | Penalità di ripetizione (es. 1.1) |
| `--rep-context N` | Finestra in token della penalità (default 20) |
| `--seed S` | Seme casuale, identico su tutti i rank (default 0) |
| `--max-kv-size N` | Tetto alla KV cache in token; oltre, finestra scorrevole |
| `--kv-bits 4\|8` | Quantizzazione della KV cache (circa ¼ o ½ della memoria rispetto a 16 bit) |
| `--kv-group-size G` | Gruppo di quantizzazione della KV cache (default 64) |
| `--quantized-kv-start N` | Quantizza la KV cache solo dopo N token (default 0) |
| `--prefill-step-size N` | Blocchi di elaborazione del prompt (default 2048) |

Esempio con campionamento e KV cache a 8 bit:

```bash
~/mlx-dist/tp.sh --model mlx-community/Qwen3.6-27B-8bit --prompt-file ~/mlx-dist/p.txt \
  --temp 0.6 --top-p 0.95 --top-k 20 --kv-bits 8 --max-tokens 2048
```

I valori 0.6 / 0.95 / 20 sono quelli raccomandati da Qwen per la modalità con ragionamento: ricordati dalla model card di Qwen3, non verificati per la 3.6.

Con `--temp > 0` tutti i rank devono campionare lo stesso token, altrimenti divergono e la generazione si blocca: per questo lo script fissa lo stesso seme ovunque.

## 10. Modelli compatibili e budget di memoria

Un modello gira in TP solo se la sua architettura (`model_type`) implementa `shard()` in mlx-lm. Le architetture supportate dalla versione installata si ricavano così:

```bash
grep -l "def shard" "$(python -c 'import mlx_lm, os; print(os.path.dirname(mlx_lm.__file__))')"/models/*.py \
  | xargs -n1 basename | sed 's/\.py$//' | sort
```

Poi confronta con il `model_type` del modello (comando `curl` della sezione 4).

Budget: circa 27 GB di memoria GPU per nodo con il limite predefinito (circa 75% di 36 GB, stima), quindi circa 50 GB tra pesi e KV cache sui due nodi; circa 55–60 GB alzando `iogpu.wired_limit_mb` (sezione 11).

Peso stimato di un modello quantizzato: parametri × (bit + 0,5) / 8 byte, dove 0,5 bit tiene conto di scale e bias di quantizzazione (gruppi da 64).

KV cache per nodo in TP (le teste KV sono divise tra i nodi; parametri in `config.json`):

```latex
M_{KV} = \frac{2 \cdot L_{att} \cdot H_{KV} \cdot d_{head} \cdot b \cdot T}{N_{nodi}}
```

Dove 2 = chiavi e valori, L\_att = strati ad attenzione completa, H\_KV = teste KV, d\_head = dimensione di una testa, b = byte per valore (2 a 16 bit), T = token di contesto. Il risultato è in byte.

Candidati da verificare con i due comandi (nomi e architetture a memoria, tranne Qwen3-32B-8bit controllato con `curl`):

| Modello (mlx-community/…) | model\_type | Peso stimato (GB) | Note |
| --- | --- | --- | --- |
| Llama-3.3-70B-Instruct-4bit | llama | 40 | Denso; architettura dell'esempio TP ufficiale MLX |
| Qwen3-Next-80B-A3B-Instruct-4bit | qwen3\_next | 45 | MoE, memoria al limite |
| Qwen3-32B-8bit | qwen3 | 35 | Denso |
| Qwen3-Coder-30B-A3B-Instruct-8bit | qwen3\_moe | 32 | MoE per codice |
| Qwen3.6-27B-8bit | qwen3\_5 | 29 | Verificato in TP, 22,4 tok/s |

I modelli densi traggono più vantaggio dal TP dei MoE con pochi parametri attivi.

Gemma: nessuna architettura Gemma implementa `shard()` in mlx-lm 0.31.3. Alternative: aggiornare mlx-lm e ripetere il `grep`; usare un solo Mac (Gemma 3 27B a 4 bit circa 15 GB, Gemma 4 31B a 4 bit circa 17 GB); oppure una patch `shard()` sul modello di Llama. Alcune conversioni Gemma 4 (`gemma4_unified`) non si caricano nemmeno su un nodo singolo con mlx-lm 0.31.3.

## 11. Ottimizzazione e risoluzione problemi

Più memoria GPU, su entrambi i nodi (non persiste al riavvio; 30000 MB lascia circa 6 GB a macOS, valore prudente stimato):

```bash
sudo sysctl iogpu.wired_limit_mb=30000
```

Processi rimasti attivi dopo un crash, da pulire prima di rilanciare:

```bash
pkill -f tp_generate.py; ssh 192.168.20.2 'pkill -f tp_generate.py'
```

| Sintomo | Causa | Soluzione |
| --- | --- | --- |
| `Permission denied` / `cannot execute` | Script lanciato come comando dopo `--` | Percorso assoluto di Python dopo `--`, o `tp.sh` |
| `cat None … non-zero exit status 1` | Effetto collaterale della chiusura dopo un altro errore | Ignorare, risolvere il primo errore |
| `No module named 'mlx_lm.examples'` | Esempi non inclusi nel pacchetto pip | Usare `tp_generate.py` |
| `exec: mlx.launch: not found` | Riga di comando corrotta nel copia-incolla | Usare `tp.sh` |
| `errno 22` (RTR) o `errno 60` all'avvio | Selezione errata del GID RDMA su Thunderbolt (bug noto, PR mlx #3468) | Aggiornare mlx su entrambi i nodi |
| Blocco con prompt lunghi, senza errori | Prefill a più blocchi in TP (segnalato su modelli grandi) | `--prefill-step-size` ≥ lunghezza del prompt |
| Blocco con `--temp > 0` | Rank che campionano token diversi | Stesso `--seed` (già impostato dallo script) |
| Errore `-12` in Send/Recv, `ibv_devinfo` bloccato | Stato RDMA incoerente (segnalato su cluster a 4 nodi) | Riavviare entrambi i Mac |
| HF: "Invalid username or password" | Repo inesistente o gated | `curl` con token (sezione 4), controllare il nome |
| Memoria esaurita | Pesi + KV cache oltre il limite | `iogpu.wired_limit_mb`, `--kv-bits 8`, `--max-kv-size` |

## 12. Glossario e riferimenti

| Sigla | Significato |
| --- | --- |
| RDMA | Remote Direct Memory Access: trasferimento diretto tra memorie dei due Mac senza passare dallo stack TCP/IP |
| JACCL | Libreria di comunicazione collettiva di Apple usata da MLX per RDMA su Thunderbolt |
| TP | Tensor Parallelism: ogni nodo tiene tutti gli strati, ma con i pesi divisi |
| Rank | Indice del processo nel gruppo distribuito (0 = coordinatore) |
| KV cache | Memoria delle chiavi/valori di attenzione: conserva il contesto già elaborato |
| MoE | Mixture of Experts: per ogni token lavora solo una parte dei pesi |
| MTU | Dimensione massima di un pacchetto di rete (9000 = jumbo frame) |
| GID | Identificativo globale di una porta RDMA |
| HF | Hugging Face |

Riferimenti (letti tramite ricerca, non verificati in dettaglio):

- [MLX — Distributed Communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [MLX — Launching Distributed Programs](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html)
- [WWDC26 — Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2026/233/)
- [mlx PR #3468 — GID su Thunderbolt RDMA](https://github.com/ml-explore/mlx/pull/3468)
- [mlx discussion #3939 — prefill bloccato in TP](https://github.com/ml-explore/mlx/discussions/3939)
- [mlx discussion #4247 — errore -12 in Send/Recv](https://github.com/ml-explore/mlx/discussions/4247)
