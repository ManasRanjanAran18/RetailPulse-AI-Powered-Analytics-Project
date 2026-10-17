# Deployment guide: GitHub → Render

Total time: about 10 minutes. You need a free [GitHub](https://github.com) account and a free [Render](https://render.com) account.

---

## Part 1 - Push the project to GitHub

### 1. Create an empty repository
1. Go to <https://github.com/new>.
2. **Repository name:** `retailpulse-analytics`
3. Choose Public or Private.
4. **Do NOT** tick "Add a README", ".gitignore" or "license" (the project already has them).
5. Click **Create repository**.

### 2. Unzip and push
Unzip `retailpulse-analytics.zip`, open a terminal (PowerShell / Terminal / VS Code terminal) **inside the unzipped
`retailpulse-analytics-main` folder** (the one containing `Dockerfile` and `README.md`), then run:

```bash
git init
git add .
git commit -m "Initial commit: RetailPulse analytics platform"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/retailpulse-analytics.git
git push -u origin main
```

> Replace `YOUR_GITHUB_USERNAME`. If asked to log in, use your GitHub username and a
> [Personal Access Token](https://github.com/settings/tokens) as the password (or sign in via the browser prompt / GitHub Desktop).

First git use on this machine? Run once:
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 3. (Optional) Replace the placeholders
Search the repo for `YOUR_GITHUB_USERNAME` (README badge, `k8s/*.yaml`) and put your username in.

GitHub Actions starts automatically: open the **Actions** tab → *CI* should turn green (tests, light pipeline, dashboard smoke test, Docker build).

---

## Part 2 - Deploy on Render

### Option A (recommended): Blueprint, one click
1. Log in at <https://dashboard.render.com>.
2. **New +** → **Blueprint**.
3. **Connect** your GitHub account and select the `retailpulse-analytics` repository.
4. Render reads `render.yaml` and shows the service `retailpulse-dashboard`. Click **Apply**.
5. Wait for the build (about 3-6 minutes the first time). Watch the logs until you see `You can now view your Streamlit app`.
6. Open the URL shown at the top, e.g. `https://retailpulse-dashboard.onrender.com`. 🎉

### Option B: manual Web Service
1. **New +** → **Web Service** → connect the repository.
2. Fill in:

| Field | Value |
|---|---|
| Name | `retailpulse-dashboard` |
| Language / Runtime | **Docker** |
| Branch | `main` |
| Dockerfile Path | `./Dockerfile` |
| Instance Type | **Free** (or higher) |
| Health Check Path | `/_stcore/health` (under *Advanced*) |

3. Click **Create Web Service**. No start command is needed; the Dockerfile handles it and binds to Render's `$PORT` automatically.

### Option C: without Docker (native Python runtime)
| Field | Value |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0` |
| Environment variable | `PYTHON_VERSION` = `3.12.3` (the pinned packages need Python ≥ 3.11) |

### Auto-deploy
Every `git push` to `main` redeploys automatically (`autoDeploy: true`).

---

## Free-tier things to know
- **Cold starts:** free services sleep after ~15 min of inactivity; the first visit afterwards takes ~30-60 s to wake up.
- **Memory:** the free tier has 512 MB RAM. The dashboard peaks at about 280 MB, which is fine. Upgrade the plan if you add much larger datasets.
- **Training is not done on Render.** Models and results are pre-computed and committed (`models/`, `data/processed/`). To refresh them, retrain locally (`python scripts/run_pipeline.py`) and `git push`. If the artifacts are ever missing, the app rebuilds a lean version of them on first launch.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No open ports detected" / port scan timeout | Make sure you use the repo's `Dockerfile` unchanged (it binds `0.0.0.0:$PORT`), or the Start Command from Option C. |
| Service restarts / "Out of memory" | Switch to a paid instance type with more RAM (also removes free-tier sleeping). |
| Build fails at `pip install` | Check *Python version* (Option C) is 3.11+; Docker option pins this for you. |
| Page loads but charts are empty | Confirm `data/processed/` was pushed (`git ls-files data/processed | head`). |
| Blank page behind a custom proxy | Streamlit needs WebSocket support; Render supports it by default. |

---

## Extra: Docker, MLflow, Kubernetes

```bash
# Local container (same image Render builds)
docker build -t retailpulse .
docker run --rm -p 8501:8501 retailpulse          # http://localhost:8501

# Dashboard + MLflow UI together
docker compose -f docker/docker-compose.yml up --build dashboard mlflow
#   dashboard http://localhost:8501   mlflow http://localhost:5000
# Retrain inside a container and log to that MLflow
docker compose -f docker/docker-compose.yml --profile train run --rm trainer

# Kubernetes (edit image names + ingress host first - see k8s/README.md)
kubectl apply -f k8s/
```

The workflow `.github/workflows/publish-image.yml` pushes images to GitHub Container Registry
(`ghcr.io/<you>/retailpulse-analytics`, `-mlflow`, `-train`) on every push to `main`, which the Kubernetes manifests use.
