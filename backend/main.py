from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
import logging

from services import PlagiarismService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LearnWise Plagiarism API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

plagiarism_service = PlagiarismService()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB per file
MAX_FILES = 50


@app.get("/")
def root():
    return {"message": "LearnWise Plagiarism API v2.0 is running"}


@app.get("/api/model-status")
def model_status():
    """Cek apakah Model A dan Model B sudah tersedia."""
    avail_a = plagiarism_service.model_a.is_available
    avail_b = plagiarism_service.model_b.is_available
    return {
        "model_a_available": avail_a,
        "model_b_available": avail_b,
        "message_a": "Model A siap digunakan." if avail_a else plagiarism_service.model_a._load_error,
        "message_b": "Model B siap digunakan." if avail_b else plagiarism_service.model_b._load_error
    }


@app.post("/api/analyze")
async def analyze_documents(
    files: List[UploadFile] = File(...),
    model: Optional[str] = Form("model_a"),
    weight_a: Optional[float] = Form(0.5),
    weight_b: Optional[float] = Form(0.5)
):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Minimal 2 dokumen diperlukan")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maksimal {MAX_FILES} dokumen diperbolehkan")

    valid_models = ["model_a", "model_b", "model_c"]
    if model not in valid_models:
        model = "model_a"

    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain"
    ]

    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' tidak didukung. Hanya PDF, DOCX, dan TXT."
            )

    try:
        file_contents = []
        for file in files:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{file.filename}' melebihi ukuran maksimal 20MB."
                )
            file_contents.append({
                "name": file.filename,
                "content": content,
                "content_type": file.content_type
            })

        result = plagiarism_service.analyze(
            file_contents,
            model=model,
            weight_a=weight_a,
            weight_b=weight_b
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat menganalisis dokumen. Pastikan semua file dapat dibaca.")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
