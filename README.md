# [PaperExplainAgent](https://mihirballari.github.io/PaperExplainAgent/)

PaperExplainAgent is a PDF-to-video explainer pipeline for STEM papers. You upload a PDF and provide an LLM API key, and the system generates a short narrated explainer video by planning scenes, generating Manim code, and rendering an MP4.

## What it does
- Converts a PDF into structured text + extracted figures
- Plans a multi-scene explanation (3–7 scenes)
- Generates Manim animation code + narration
- Renders and returns a final MP4 (plus intermediate artifacts)

## Repo structure
- `frontend/` — React demo UI (PDF upload, status updates, output preview)
- `backend/` — FastAPI server (job orchestration + status endpoints)
- `theorem_explain_agent/` — generation pipeline (planning → codegen → render)

## Demos
- Live demo UI: https://math-495-final-a5d936bb1f71.herokuapp.com/#
- Project page: https://mihirballari.github.io/PaperExplainAgent/

## Local run 
1. Start backend (FastAPI)
2. Start frontend (React)
3. Paste API key, upload PDF, click **Generate**, and wait for the job to finish

## Notes
Generation is compute-heavy and can take several minutes depending on PDF length and model settings.
