# SAM3 Annotation Studio

Manual image segmentation tool for producing COCO datasets for SAM3 fine-tuning.

## Quick start

```bash
./start_all.sh
```

Opens:
- Frontend: http://127.0.0.1:5173
- Backend API: http://127.0.0.1:8002
- Tiled: http://127.0.0.1:8010

## Tabs
1. **Connect** — pick a Tiled dataset or local folder
2. **Annotate** — draw shapes (polygon, rectangle, ellipse, brush, eraser), manage classes, navigate slices
3. **Export** — write COCO dataset for SAM3 fine-tuning

## Output format
COCO JSON adapted for SAM3: `segmentation` is always compressed RLE, `categories[].name` is the SAM3 concept phrase.

## Development

```bash
# Backend
cd backend && pip install -e ".[dev,test]"
pytest

# Frontend
cd frontend && npm install
npm test
npm run dev
```
