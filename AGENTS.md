# AGENTS.md — Segmentation Annotation Studio

## Stack
- Backend: FastAPI, Python ≥ 3.11, pyproject.toml, pytest
- Data: Tiled (local SQLite catalog for dev)
- Frontend: Vite, React, TypeScript, react-konva, Zustand (+ zundo temporal middleware), TanStack Query, Tailwind CSS

## ALS Standards
- Python: full type annotations, Google-style docstrings, pathlib.Path, no print()
- React: Finch shell (HubAppLayout/HubSidebar/HubHeader), Tailwind CSS, Zustand stores
- Tiled: all Tiled access goes through the backend API — frontend never calls Tiled directly
- Icons: @phosphor-icons/react for all UI icons

## Security rules
- NEVER expose Tiled API keys to the frontend (no api_key in /api/config/servers response)
- NEVER accept server_api_key query params from clients — resolve keys server-side via api_key_for_uri()
- NEVER bind services to 0.0.0.0 locally — always 127.0.0.1

## Tiled safety rules
- NEVER call Tiled write endpoints without explicit user confirmation
- ALL Tiled access from frontend goes through backend API (port 8002)
- TILED_API_KEY is read from backend/.env — never hardcoded

## Annotation data model (do NOT change without updating both backend/schemas.py and frontend stores)
- All shape coordinates are IMAGE pixels (Stage transform is display-only)
- BrushShape.strokes is ordered: paint strokes OR pixels in, erase strokes AND NOT out
- rectangle/ellipse are stored NORMALIZED: w,h,rx,ry >= 0

## Secrets
- .env is git-ignored — never commit it
- Add new secrets to .env.example with placeholder values immediately

## Testing rules
- Write test structure BEFORE implementing the feature (TDD)
- Tests must test behavior/interface, NOT implementation details
- Backend integration tests in tests/
- Frontend: Vitest unit tests per component in component subfolder

## Git
- Commit after every working increment
- Format: <type>(<scope>): <description>

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
