"""
Document routes: upload, list, and delete medical education documents.
"""
import os
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, BackgroundTasks
from loguru import logger

from backend.api.schemas import UploadResponse, DeleteResponse, DocumentInfo
from backend.security.auth import get_current_user, validate_file_extension, validate_file_size
from backend.rag.document_processor import process_document
from backend.vectorstore.pinecone_store import index_document, get_vector_store
from backend.utils.file_utils import save_uploaded_file, delete_uploaded_file

router = APIRouter(prefix="/documents", tags=["Documents"])

# In-memory document registry (replace with DB in production)
_document_registry: dict[str, dict] = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    """
    Upload and index a medical document (PDF, DOCX, or TXT).
    Pipeline: Validate → Extract → Chunk → Embed → Index in Pinecone
    """
    # Validate file extension
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF, DOCX, and TXT files are allowed.",
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if not validate_file_size(len(content)):
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {50} MB.",
        )

    logger.info(f"Uploading document: {file.filename!r} ({len(content)} bytes) by user={current_user!r}")

    # Save to disk
    file_path, doc_id = save_uploaded_file(content, file.filename)

    try:
        # Process document (extract + chunk)
        processed = process_document(file_path=file_path, filename=file.filename, doc_id=doc_id)

        # Index into vector store
        indexed_count = index_document(processed.chunks)

        # Register document
        _document_registry[doc_id] = {
            "doc_id": doc_id,
            "filename": file.filename,
            "file_path": file_path,
            "chunk_count": indexed_count,
            "uploaded_by": current_user,
        }

        logger.info(f"Document indexed: doc_id={doc_id}, chunks={indexed_count}")
        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            total_chunks=indexed_count,
            message=f"Document '{file.filename}' successfully processed and indexed with {indexed_count} chunks.",
        )

    except Exception as e:
        # Clean up on error
        delete_uploaded_file(file_path)
        logger.error(f"Document processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.get("/list", response_model=list[DocumentInfo])
async def list_documents(
    current_user: str = Depends(get_current_user),
):
    """List all indexed documents."""
    docs = []
    for doc_id, info in _document_registry.items():
        docs.append(
            DocumentInfo(
                doc_id=doc_id,
                filename=info["filename"],
                chunk_count=info.get("chunk_count", 0),
            )
        )
    return docs


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(
    doc_id: str,
    current_user: str = Depends(get_current_user),
):
    """Delete a document and all its indexed chunks."""
    if doc_id not in _document_registry:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    info = _document_registry[doc_id]

    # Delete from vector store
    store = get_vector_store()
    store.delete_by_doc_id(doc_id)

    # Delete file from disk
    delete_uploaded_file(info.get("file_path", ""))

    # Remove from registry
    del _document_registry[doc_id]

    logger.info(f"Document deleted: doc_id={doc_id} by user={current_user!r}")
    return DeleteResponse(
        doc_id=doc_id,
        message=f"Document '{info['filename']}' deleted successfully.",
    )
