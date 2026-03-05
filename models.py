from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    prompt: str
    top_k: int = 10  # how many relevant files to pull into context


class OperationRecord(BaseModel):
    operation: str       # create | update | append | read | error
    path: Optional[str] = None
    tool: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    files_referenced: List[str]
    files_modified: List[str]
    operations_performed: List[OperationRecord]


class IndexStatus(BaseModel):
    vault_path: str
    total_files: int
    indexed_files: int
    last_indexed: Optional[str] = None
    watching: bool = False


class FileSearchResult(BaseModel):
    path: str
    score: float


class FileListResponse(BaseModel):
    files: List[FileSearchResult]
