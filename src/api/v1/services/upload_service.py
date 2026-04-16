import shutil
from pathlib import Path
from fastapi import UploadFile
from src.ingestion.ingestion import ingest_pdf


def process_and_ingest_document(file: UploadFile) -> dict:
    base_dir = Path(__file__).resolve().parents[4]
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / file.filename
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    document_id = ingest_pdf(str(file_path))
    return {"filename": file.filename, "document_id": document_id}
