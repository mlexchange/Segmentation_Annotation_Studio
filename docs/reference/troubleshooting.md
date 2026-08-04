# Troubleshooting

Common issues and how to resolve them.

## The app won't start

??? question "`npm / Node.js 20+ was not found on PATH`"
    Install Node.js 20 or newer and make sure `node` and `npm` are on your
    `PATH`, then re-run `./start_all.sh`. See [Installation](../getting-started/installation.md#prerequisites).

??? question "A port is already in use"
    The launcher scans upward for a free port automatically. To pin ports,
    launch with overrides:

    ```bash
    FRONTEND_PORT=5200 BACKEND_PORT=8100 TILED_PORT=8110 ./start_all.sh
    ```

??? question "The frontend loads but nothing connects"
    Check the backend is healthy:

    ```bash
    curl http://127.0.0.1:8002/health
    ```

    The Vite dev server proxies `/api` to `http://127.0.0.1:8002`. If you
    changed the backend port, set `API_PROXY_TARGET` accordingly.

## Connecting to data

??? question "Tiled connection fails or shows 0 samples"
    - Confirm `TILED_URI` in `backend/.env` points at a running Tiled server.
    - Make sure `TILED_API_KEY` is set (the launcher generates one on first run).
    - Use **Re-verify** on the Connect tab after fixing settings.

??? question "Local folder shows no images"
    Only **TIFF, PNG, JPG, and NPY** files are recognized. Confirm you granted an
    **absolute** path and selected the folder that directly contains the images.

## The Smart (AI) tool

??? question "'Smart (AI)' is disabled or stuck loading"
    - The SAM model downloads in the background on first launch; give it a moment.
    - If your browser lacks WebGPU it falls back to CPU (slower) — the status
      line shows which backend is active.
    - You can pre-fetch the model manually:

      ```bash
      cd frontend && node scripts/fetch-sam-model.mjs
      ```

??? question "SAM results look stale after changing brightness"
    Adjusting **Display** brightness/contrast re-encodes the slice for SAM. Set
    your display first, then use Smart mode.

## Exporting

??? question "'Push masks to Tiled' is greyed out"
    It only works for **Tiled** sources — it's disabled for local folders. Use
    **Export → Download .zip** to get data from a local source.

??? question "No 'Download .zip' button appears"
    The download button only shows for COCO **Export** jobs, not for **Push masks
    to Tiled** jobs (which write to Tiled, not to a file).

??? question "Exported images look different from the canvas"
    Exported PNGs bake in your **Display** render settings (brightness, contrast,
    colormap, gamma). Adjust those before exporting if needed.

## Saving

??? question "Did I lose my work?"
    The app autosaves a local draft roughly every 1.5 seconds for crash recovery.
    For a durable, restorable snapshot, use **Save → Save version**. Recover
    earlier states from **Version History**.

---

Still stuck? Check the terminal running `start_all.sh` for backend and Tiled
logs — most connection and export errors surface there.
