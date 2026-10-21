# PageGlance

PageGlance is a webpage question-answering project with a Chrome extension frontend and a separate FastAPI backend.

## Project structure

- `extension/` — React, TypeScript, Vite, and ESLint Chrome extension.
- `backend/` — FastAPI service with Pydantic request validation and Ruff configuration.

## Windows PowerShell setup

Run these commands from the repository root.

### Frontend

```powershell
Set-Location extension
npm install
npm run lint
npm test
npm run build
```

The build creates the unpacked Chrome extension in `extension\dist\`. To point it at a local API, copy `.env.example` to `.env.local` inside `extension\` and set `VITE_PAGEGLANCE_API_URL` before building. The default is `http://127.0.0.1:8000`. This setting contains an address only; do not put API keys or other secrets in frontend environment variables. The manifest grants local development host access to `localhost` and `127.0.0.1`.

To load the extension in Chrome:

1. Open `chrome://extensions`.
2. Turn on **Developer mode**.
3. Choose **Load unpacked** and select the repository's `extension\dist` directory.
4. Open a normal webpage, select the PageGlance extension icon, enter a question, and choose **Ask PageGlance**.

Use the expand icon in the popup header to open the same page conversation in a dedicated PageGlance tab. Messages and drafts are restored from extension storage, and the source webpage is re-read in the new tab. The `chat.html` entry is included in the same `dist` build. The interface uses a subtle animated gradient that stops moving when reduced motion is enabled.

Chrome restricts extensions on some pages, including `chrome://` pages and the Chrome Web Store. PageGlance reports when Chrome prevents extraction. The backend must be running at the configured URL.

The popup's development server can be started with `npm run dev` from `extension\`, but Chrome must load the built `dist\` directory for extension behavior.

### Backend

```powershell
Set-Location backend
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item ..\.env.example ..\.env
notepad ..\.env
ruff check .
ruff format --check .
pytest
uvicorn app.main:app --reload
```

If PowerShell blocks virtual environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again. The API is available at `http://127.0.0.1:8000`; interactive docs are at `/docs`.

In the root `.env` file, replace `your_groq_api_key_here` with a Groq API key from [Groq Console](https://console.groq.com/keys). Keep this key in the backend environment only; never put it in `extension/.env.local`. The default model is `openai/gpt-oss-20b`. Set `GROQ_MODEL` in `.env` to another currently available Groq chat model if needed. The timeout and completion token limit can also be adjusted with `GROQ_TIMEOUT_SECONDS` and `GROQ_MAX_COMPLETION_TOKENS` (allowed ranges are 1–120 seconds and 1–4096 tokens).

`PAGEGLANCE_CONTEXT_MAX_CHARS` limits the page text sent to the language model (default `16000`, allowed range `1000`–`100000`). Longer pages keep the introduction and passages most relevant to the question; the backend logs only the original and selected character counts. Character count is a conservative proxy for tokens, so keep this limit modest to leave room for the prompt and your Groq account's input-token rate limit. This avoids a simple prefix-only cutoff.

Follow-up context is limited by `PAGEGLANCE_HISTORY_MAX_MESSAGES` (default `10`, allowed range `0`–`20`) and `PAGEGLANCE_HISTORY_MAX_CHARS` (default `12000`, allowed range `1000`–`40000`). `POST /ask` and `POST /ask/stream` accept optional recent `history` messages; existing callers can omit the field.

`GET /health` reports service status. `POST /ask` accepts a question and bounded page context (`title`, `url`, and readable `text`), then returns `{"answer": "..."}` generated from the supplied page. `POST /ask/stream` is used by the popup to stream answer deltas. Both accept optional bounded `history` messages for follow-up questions; existing callers can omit that field. Page text is treated as untrusted data, and the model is instructed to ignore any directives embedded in it and acknowledge when the page does not support an answer. Provider credentials remain on the backend.

The popup saves conversations and drafts locally per normalized page URL. A restored answer marked as interrupted was still streaming when its popup closed; reopening does not repeat the request. Welcome suggestions submit ordinary questions through the same extraction and streaming flow.

### Automatic Windows development startup

The optional per-user Windows Task Scheduler task starts PageGlance after you sign in. From the repository root, run this once in PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-pageglance-startup.ps1
```

The installer reuses a working backend virtual environment or the currently available Python runtime when its dependencies are installed; otherwise it creates `backend\.venv` and installs the dependencies declared in `backend\pyproject.toml`. It runs `npm ci` only if the local Vite installation is missing. The task starts FastAPI, waits up to 45 seconds for `GET /health` to return `{"status":"ok"}`, then starts Vite on `127.0.0.1:5173`, waits up to 30 seconds for its development page, and opens that page in the default browser. Existing healthy services are reused. Unexpected port occupants, URL mismatches, missing dependencies, and timeouts are logged and stop startup without retrying forever.

The startup script reads `PAGEGLANCE_HOST` and `PAGEGLANCE_PORT` from the root `.env` (defaults `127.0.0.1` and `8000`). It resolves `VITE_PAGEGLANCE_API_URL` from the process environment or the extension's Vite environment files and checks that it targets the same loopback backend. When the setting is not explicitly present, it supplies the computed local API URL to that Vite process without changing any `.env` file. The Chrome extension build keeps its existing Vite configuration and URL behavior.

To start both services manually, run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-pageglance.ps1`. To stop only processes started by this script, run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop-pageglance.ps1`; processes without a matching recorded PID and executable are left alone. To disable automatic startup while keeping the services running, run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\uninstall-pageglance-startup.ps1`. Add `-StopServices` to that command to also stop the tracked services. You can also disable or remove **PageGlance Development Services** in Windows Task Scheduler. Logs and PID records are stored under `%LOCALAPPDATA%\PageGlance\`, with separate backend/frontend output logs in its `logs` subdirectory.
