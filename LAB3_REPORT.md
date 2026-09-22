# Lab 3 — Containerizing the Model with Docker

**Course:** MLOps
**Repo:** [github.com/Georges-haj/mlops-lab-1](https://github.com/Georges-haj/mlops-lab-1)
**Continues from:** Lab 2 (training + MLflow tracking)

This one took a lot longer than the Dockerfile itself would suggest. The actual multi-stage build came together fine — what ate the time was a slow connection (package installs that should take a minute took 25+) and a real networking/storage bug once I tried to connect the container to my Lab 2 MLflow server. Writing that part up in detail because it's genuinely what I learned from this lab, not the Dockerfile syntax.

## Registering the model

Used my best Lab 2 run (`gaudy-crab-53`, ID `52a9649eb37b4fa78a62147cceaf9a44`, `val_accuracy=0.717`):

```python
mlflow.register_model('runs:/52a9649eb37b4fa78a62147cceaf9a44/model', 'food11')
client.set_registered_model_alias('food11', 'champion', 1)
```

**Q1 — version number, and run artifact vs. registered model:** Version **1**. A run's logged model artifact is a file tied permanently to that one training execution. A registered model is a separate object with its own version history — `food11` v1 points at this run now, but v2 could point at something else entirely later. The registry is the indirection layer that makes "the model" a stable reference.

**Q2 — aliases vs. stages:** `Staging`/`Production` are the deprecated built-in stages; **aliases** replace them. You version separately from the run because a run is an immutable record of one training execution — you don't want "what experiment happened" tangled with "what's currently serving." An alias beats a fixed stage name because `Production` was one global slot baked into MLflow itself; an alias is just a string I define, and I can have as many (`champion`, `challenger`, `canary`) as I want, each reassignable independently.

## Serving script

`src/food11/serve.py` loads via `mlflow.pyfunc.load_model("models:/food11@champion")`, exposes `/health` and `/predict`. Loads the model once at startup (`lifespan` handler, not per-request — too slow otherwise), reuses `CATEGORIES`/normalization constants from the Lab 1/2 files instead of retyping them, and applies softmax at the end since `CrossEntropyLoss` trained on raw logits, not probabilities.

**Q3 — why a model URI instead of a raw .pth file:** A raw file path hardcodes one exact run's location into the serving code — switching models means editing and redeploying the code. The URI means the code never changes, it always asks for "whatever `champion` points to now." Also a `.pth` only has weights, not architecture — the code would separately need to rebuild the exact `resnet18` shape before loading weights in; `mlflow.pyfunc.load_model` reconstructs the whole packaged thing on its own. To serve a newer version: reassign the alias, restart. Nothing in `serve.py` changes.

Tested locally first, got 4/5 correct on validation images — consistent with the 75.8% test accuracy from Lab 2, not a bug.

## The Dockerfile

Multi-stage: a `builder` stage with `uv` and full dependency resolution, a slim `runtime` stage that only copies the finished `.venv` and `src/`. Two separate `uv sync` calls — one right after copying `pyproject.toml`/`uv.lock`, one after `src/` is copied in.

**Q4 — why manifests before source:** As long as `pyproject.toml`/`uv.lock` don't change, Docker reuses the cached result of the first (expensive) `uv sync` on rebuild. Confirmed for real: the second `uv sync`, after `src/` was copied in, finished in **0.4 seconds** (`Checked 97 packages in 1ms`) versus the first one's **27 minutes**.

**Q5 — naive vs. multi-stage size:** Built a naive single-stage version to compare for real.

| Image | Size |
|---|---|
| naive | 2.12GB |
| multi-stage | 1.99GB |

**~130MB saved.** `docker history` shows why: the naive image has an extra 55.8MB layer that's just the `uv` binary itself, still present even though nothing needs it at runtime, plus a larger dependency layer (1.48GB vs 1.43GB) from `uv`'s own leftover download cache with nowhere to be discarded. Same root cause both times — a single-stage build keeps everything used *during* the build, forever.

**Q6 — missing .dockerignore:** Given my Dockerfile only does selective `COPY pyproject.toml uv.lock ./` and `COPY src/ ./src/` (never `COPY . .`), nothing would actually break — but `docker build` still tars up and sends the entire directory to the daemon before building starts, regardless of what gets copied. In my repo that's `data/` (1.27GB) + `data_local/` (1.3GB) + `.git/` — multiple gigabytes re-uploaded on every build, even a one-line `serve.py` edit. The one folder that *would* actually break a build if a lazier Dockerfile used `COPY . .`: `.venv/` — a locally-built virtualenv has Windows/Mac-compiled binaries, and copying that into a Linux container causes real import failures.

## Running it — where the actual debugging happened

`docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest`

**Q7 — why not 127.0.0.1, and what host.docker.internal resolves to:** `127.0.0.1` means "myself," but from the perspective of whoever's asking — a container has its own network namespace, so inside the container it means the container, not my Windows machine. `host.docker.internal` is a Docker Desktop-specific DNS name (Mac/Windows only, which is why the lab gives Linux `--network host` instead) that resolves back to whatever address reaches the actual host from inside the container.

That's the textbook answer. What I actually hit going beyond it:

- **Reachable but rejected.** Got `403: Invalid Host header - possible DNS rebinding attack detected`. MLflow 3.x only accepts requests whose `Host` header matches an allowlist (default: localhost + private IPs); `host.docker.internal` isn't on it. Fixed with explicit `--allowed-hosts host.docker.internal:5000,...` (a bare `"*"` got shell-glob-expanded into every filename in my project directory — had to avoid that).
- **Reachable, accepted, still failed.** Model loading then crashed with `No such artifact: ''`. The stored artifact location was `file:D:/Georges/.../mlruns/...` — a hardcoded Windows path baked into the DB at logging time. A local-filesystem artifact store only works for a process with direct disk access to that exact path; a container's filesystem is separate, full stop, no networking fix helps. Reconfiguring the server to default to MLflow's proxied `mlflow-artifacts:/` scheme didn't help either — an experiment's artifact root is fixed at creation time, and `food11` already existed from Lab 2. Working fix: since `D:/Georges/...` doesn't start with `/`, Linux treats it as *relative*, not absolute — bind-mounted the host's `mlruns` folder into the container at that exact relative path under `/app`, and it resolved. This is a one-off workaround for an already-logged model's storage location, not a general pattern — the real fix is never using a bare local path as an artifact root to begin with.

**Q8 — stop and restart from the same image:** Removed the container, started a fresh one from the same image, no rebuild. Same prediction, same confidence, immediately. Confirms the image only ever contains **code** — never the trained weights. Every container start re-fetches whatever `champion` currently points to. The image and the model version are fully independent; I could reassign `champion` to a different model and restart this same image with zero rebuild.

## Committing

```bash
git add Dockerfile .dockerignore src/food11/serve.py pyproject.toml uv.lock
git commit -m "Containerize model serving with Docker"
git push
```

Also added `__pycache__/` to `.gitignore` — it had never actually been excluded before now.

**Q9 — what's still missing for another machine to run this:** The Dockerfile is versioned by git; the actual built image only exists on my laptop's Docker daemon and was never pushed anywhere. Still needed:
1. A **container registry** to push to (Docker Hub, GHCR, ECR) — same role DagsHub played for DVC data.
2. An **immutable tag** — `:latest` is a moving target; need a commit SHA or digest so "the exact image" is unambiguous.
3. A **CI pipeline** building and pushing per commit, not a human running `docker build` by hand.
4. A tracking server that's an actual network-reachable service with a real artifact store — not SQLite plus a local path on one laptop. A CI runner would hit exactly what I hit, minus `host.docker.internal` to even partly paper over it.

## Takeaway

The Dockerfile syntax was the easy part. The real work was the gap between "the container can reach a network address" and "the container can actually get what it needs" — those turned out to be three separate failures stacked on each other, each with a completely unrelated-looking error message. Containerizing something is the easy 80%; what it depends on outside the image is the actual hard part.
