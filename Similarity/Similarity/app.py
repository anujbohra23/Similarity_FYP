from fastapi import FastAPI, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from audio_pipeline import transcribe_and_embed
from image_pipeline import caption_and_embed
from similarity_utils import compute_cosine_similarity

app = FastAPI()

# Serve HTML frontend
app.mount("/static", StaticFiles(directory="templates"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/audio/similarity")
async def audio_sim(file1: UploadFile, file2: UploadFile):
    transcript1, embedding1 = transcribe_and_embed(file1.file)
    transcript2, embedding2 = transcribe_and_embed(file2.file)
    similarity = compute_cosine_similarity(embedding1, embedding2)
    return {
        "transcript1": transcript1,
        "transcript2": transcript2,
        "embedding1": embedding1.tolist(),
        "embedding2": embedding2.tolist(),
        "cosine_similarity": similarity,
    }


@app.post("/image/similarity")
async def image_sim(file1: UploadFile, file2: UploadFile):
    caption1, embedding1 = caption_and_embed(file1.file)
    caption2, embedding2 = caption_and_embed(file2.file)
    similarity = compute_cosine_similarity(embedding1, embedding2)
    return {
        "caption1": caption1,
        "caption2": caption2,
        "embedding1": embedding1.tolist(),
        "embedding2": embedding2.tolist(),
        "cosine_similarity": similarity,
    }
