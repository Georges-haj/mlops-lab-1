# Lab 2 — Model Training and Experiment Tracking with MLflow

**Course:** MLOps
**Repo:** [github.com/Georges-haj/mlops-lab-1](https://github.com/Georges-haj/mlops-lab-1)
**Continues from:** Lab 1 (git/dvc + Food-11 data prep)

## A quick note before I start

This one builds directly on Lab 1 — same repo, same dataset, now actually training something with it. I'd never touched MLflow before this, so a lot of this report is me hitting an error, reading the traceback properly instead of panicking, and figuring out what it actually meant. I'm including those bumps rather than pretending the first attempt at everything just worked, because honestly it didn't, and I think the mistakes taught me more than the parts that went smoothly.

## Environment setup

The install step (`uv add mlflow torch torchvision scikit-learn`, with the CPU-only PyTorch index trick in `pyproject.toml`) was actually already done — my professor gave us this exact instruction at the end of Lab 1's session as prep for this lab, so by the time I started Lab 2 it was just a matter of confirming it was still there.

### Question 1 — what changed in pyproject.toml and uv.lock?

`pyproject.toml`'s `dependencies` list gained `mlflow`, `torch`, `torchvision`, and `scikit-learn`, plus a `[[tool.uv.index]]` block pinning `torch`/`torchvision` specifically to PyTorch's CPU-only wheel index instead of the default (CUDA-enabled) one. `uv.lock` grew by about 2,700 lines, which genuinely made me stop and go "wait, why?" — I only asked for 4 packages. Turns out the lockfile isn't just recording what I typed, it's pinning the exact resolved version of the *entire* transitive dependency tree: numpy, pandas, scipy, flask, fastapi, protobuf, sqlalchemy, opentelemetry, pyarrow, and dozens more that MLflow alone drags in behind the scenes. I hadn't really thought about how many indirect dependencies a single package like MLflow could have until I saw the number.

### Running the tracking server

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

First time I ran this it took a genuinely long time to come up — turns out MLflow was creating the SQLite schema from scratch (`Creating initial MLflow database tables...`) and spinning up its Uvicorn workers, which on Windows apparently takes 30-40 seconds. Once I saw `Uvicorn running on http://127.0.0.1:5000` in the log I knew it was actually alive, not stuck.

Opening `http://127.0.0.1:5000` showed exactly what the lab said I'd see: an empty "Default" experiment.

### Question 2 — backend-store-uri vs default-artifact-root

- `--backend-store-uri sqlite:///mlflow.db` is where MLflow keeps **metadata**: experiments, runs, params, metrics, tags. This is structured, queryable stuff — the kind of thing you'd filter or sort, so it makes sense it lives in a real database.
- `--default-artifact-root ./mlruns` is where MLflow keeps **artifacts**: arbitrary files a run produces — the trained model itself, plots, checkpoints. You don't query these, you just fetch the whole thing, so they sit on disk as plain files instead.

I actually checked this on my own filesystem afterward: right after starting the server, `mlflow.db` existed (876KB, with the schema and the empty Default experiment row already in it) but `mlruns/` didn't exist yet at all — it only gets created the first time an actual run logs an artifact. That distinction — metadata exists from the start, artifacts are created lazily — made the concept click a lot more than just reading the definitions did.

### Question 3 — keeping mlflow.db and mlruns/ out of git and dvc

```bash
echo "mlflow.db" >> .gitignore
echo "mlruns/" >> .gitignore
git add .gitignore
git commit -m "Ignore local mlflow tracking files"
git push
```

Neither belongs in git: `mlflow.db` changes on literally every metric logged during every run, so git would be generating constant, meaningless diffs on a binary file — and it's local scratch state anyway, not something two people should be merging.

They don't belong in dvc either, and this took me a second to actually reason through rather than just accepting it: dvc is for data you're deliberately curating and want to reproduce byte-for-byte across commits (like the Food-11 images from Lab 1). `mlflow.db`/`mlruns` aren't that — they're the *output* of running experiments, constantly changing, and MLflow itself is already the right tool for tracking and versioning that specific kind of data (it has its own UI, its own querying). Pointing dvc at it too would just be redundant churn for no benefit.

### Pointing the code at the server

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("food11")
```

### Question 4 — what happens when the experiment doesn't exist yet?

I watched this happen directly in the server logs the first time I ran training:
```
INFO mlflow.tracking.fluent: Experiment with name 'food11' does not exist. Creating a new experiment.
```
No error, no need to pre-create anything through the UI — it just gets created automatically the first time `set_experiment` is called with a new name.

## Writing the training script

This was the part I was honestly most nervous about — I've written small PyTorch things before for a class exercise, but never anything wired up to an actual tracking system, and never touched `ImageFolder` before either. I wrote `src/food11/train.py` to:
1. Load `data/food11_processed_mini` (or `food11_processed` for the full run) with `torchvision.datasets.ImageFolder` — one `ImageFolder` per split (training/validation/evaluation), since our folder layout from Lab 1 already separates them that way.
2. Take a pretrained `resnet18`, swap its final layer for 11 outputs instead of 1000.
3. Accept `--dataset`, `--epochs`, `--lr`, `--batch-size` as CLI args.
4. Wrap training in `with mlflow.start_run():`, log the hyperparameters once at the start, log `train_loss`/`val_loss`/`val_accuracy` every epoch, and log the final `test_accuracy` plus the model itself at the end.

### Question 5 — log_param vs log_metric

Best I can explain it in my own words: `log_param` is for a value that's fixed for the entire run — you set it once, before training even starts, and it never changes (learning rate, batch size, which dataset). `log_metric` is for a value that's produced *during* training and keeps evolving — loss and accuracy are obviously different at epoch 1 than at epoch 5.

That's presumably why `log_metric` takes a `step` argument and `log_param` doesn't: MLflow needs to know *which point in time* a metric value belongs to so it can draw it as a curve (that's what those metric charts in the UI actually are, once I looked at one properly). A param is just one fixed fact about the run — there's no "step" for it to vary over, so there'd be nothing to plot even if you wanted to.

### First run, and the two bugs I hit

```bash
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
```

The actual training worked perfectly the first time — 5 epochs, sensible-looking loss curves, all logged correctly. It crashed right at the very last line, `mlflow.pytorch.log_model(...)`, with two separate problems stacked on top of each other:

1. This version of MLflow needs an `input_example` to log a PyTorch model (it uses it to trace the model's expected input shape). I hadn't passed one. Fix: grab one sample batch from the training loader before the loop and pass it in.
2. While MLflow was handling that first error and trying to print a friendly "🏃 View run at: ..." message to the console, it hit a **second, unrelated** crash — a `UnicodeEncodeError`, because Windows' default console encoding (cp1252) can't represent the running-person emoji. This is what actually showed up as the final error in my terminal, which was confusing until I read further up the traceback and found the *real* first exception underneath it.

Fixed both: added the `input_example`, and ran with `PYTHONUTF8=1` set so Python uses UTF-8 for console output regardless of Windows' default codepage. Second attempt ran clean end to end.

(One more thing that tripped me up mid-lab, unrelated to the code: the MLflow server itself had quietly stopped running between one work session and the next — it's a background process tied to that session, not something that survives on its own. Simple fix, just had to notice it and restart it before training would connect.)

### Question 6 — inspecting the first successful run

Opened it in the UI and found exactly what the lab describes:
- **5 params:** `dataset=mini`, `epochs=5`, `lr=0.001`, `batch_size=32`, `model=resnet18`
- **4 metrics:** `train_loss`, `val_loss`, `val_accuracy` (all logged per-epoch, so they render as charts), plus the final `test_accuracy`
- **1 logged model**, status "Ready"
- Run status: **Finished**

Where the model artifact actually lives — I went and looked directly on disk rather than just trusting the UI:
```
mlruns/1/models/m-555cd9de0f364d7da70512c1ecb9e39d/artifacts/
├── MLmodel              (metadata: flavor, signature, input example)
├── data/model.pt2       (the actual model weights/traced graph)
├── conda.yaml / python_env.yaml / requirements.txt
└── input_example.json / serving_input_example.json
```
`1` is the `food11` experiment's numeric ID. Seeing this laid out as plain files really is the metadata-vs-artifacts split from Question 2 made concrete — none of this is a database row, it's just a folder MLflow organized for me.

## Comparing hyperparameters

Ran the remaining three variations (the first training run above already covers `lr=0.001, batch_size=32`, so I didn't need to repeat it):

```bash
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.01 --batch-size 32
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.0001 --batch-size 32
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 64
```

Results across all 4 runs:

| lr | batch_size | val_accuracy (final epoch) | test_accuracy |
|---|---|---|---|
| 0.01 | 32 | 0.175 | 0.172 |
| 0.001 | 32 | 0.495 | 0.498 |
| **0.0001** | **32** | **0.717** | **0.758** |
| 0.001 | 64 | 0.547 | 0.589 |

### Question 7 — which lr won, and is higher always better?

**No, definitely not.** The lowest learning rate I tried (0.0001) gave by far the best result, and the highest (0.01) was the worst by a wide margin — its `val_loss` actually spiked to 19 in the first epoch, and accuracy never really recovered past ~0.19 across all 5 epochs. That makes sense once I thought about *why*: I'm fine-tuning a pretrained ResNet18, not training from scratch. A learning rate too high doesn't just train slowly, it actively destroys the useful features the pretrained weights already had, before the model gets a chance to adapt them gently to Food-11.

### Question 8 — parallel coordinates plot

I set the axes to `batch_size`, `lr`, and `val_accuracy` (had to actually dig into the UI a bit — it defaults to `train_loss`, and MLflow's newer version buries the classic runs table under a "Model training" toggle since it seems built GenAI-first now). The pattern was pretty visually obvious once set up correctly: the three runs at `batch_size=32` fan out across very different `lr` values, and their lines land at completely different points on the `val_accuracy` axis in a way that tracks lr almost perfectly — low lr → high accuracy, high lr → low accuracy, no crossing lines in between. The one `batch_size=64` run sits by itself and landed slightly higher on `val_accuracy` than the equivalent `batch_size=32, lr=0.001` run, but since I only tested one batch size variation, I wouldn't read much into that beyond "worth trying more batch sizes later" — the learning rate effect is the dominant, clearly visible one here.

### Question 9 — best run by val_accuracy

Sorted the runs table by `val_accuracy` descending directly in the UI. Winner:

**Run `gaudy-crab-53`, ID `52a9649eb37b4fa78a62147cceaf9a44`** — `lr=0.0001`, `batch_size=32`, `val_accuracy=0.717`, `test_accuracy=0.758`. Noted for next lab as instructed.

## Committing the training code

```bash
git add src/food11/train.py pyproject.toml uv.lock
git commit -m "Add training script with mlflow tracking"
git push
```

(`pyproject.toml`/`uv.lock` were already committed from the environment-setup step, so my actual commit just added `train.py` — plus a small `.claude/launch.json` I added for my own convenience running the tracking server, which doesn't affect the lab itself.)

## Takeaways

Going in, I assumed the "hard part" of this lab would be the MLflow API itself, and it really wasn't — `log_param`/`log_metric` are about as simple as they look, maybe 10 minutes to understand once I saw them in code. The actual hard part, and the part I didn't expect, was everything around it: figuring out why a background server had died, reading a Windows-specific Unicode crash and realizing it was hiding a completely different real error underneath, digging through a UI that didn't look like the screenshots I'd have expected. None of that is really "MLOps" in the conceptual sense, but I'm starting to suspect it kind of is the job in practice.

The part that actually taught me something about the concept itself was watching the same fine-tuning setup go from basically random guessing to genuinely learning, purely by moving the learning rate down two orders of magnitude — and having every attempt's exact settings sitting there to compare instead of me trying to remember which run used which number. That's obviously the actual problem MLflow exists to solve: not the logging itself, but not losing track of what you already tried once you're past your second or third experiment. First lab where I feel like I get *why* a tool exists, not just how to call its API.
