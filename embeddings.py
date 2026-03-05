"""
embeddings.py — Builds and queries a semantic search index over the vault.

Uses sentence-transformers (all-MiniLM-L6-v2) for local embeddings — no
external API needed.  The index is persisted to a pickle cache so it
survives restarts without re-embedding everything.

For 200–500 files this runs comfortably in memory with numpy cosine
similarity.  If your vault grows past ~2000 files, swap the search()
method to use FAISS (drop-in replacement, commented below).
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# How many characters of each file to embed.
# Keeps embedding fast; the full file is still available for context.
EMBED_CHARS = 2000


class VaultIndex:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_path: str = ".vault_index.pkl",
    ):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.cache_path = Path(cache_path)

        # Core state
        self.embeddings: Optional[np.ndarray] = None   # shape (N, dim)
        self.file_paths: list[str] = []                # vault-relative paths
        self.file_contents: dict[str, str] = {}        # path → full raw text

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, vault_files: list[dict]) -> None:
        """
        Build the index from scratch given a list of file dicts produced
        by VaultManager.read_file().
        """
        valid = [f for f in vault_files if f is not None]
        logger.info(f"Building embedding index for {len(valid)} files…")

        texts: list[str] = []
        self.file_paths = []
        self.file_contents = {}

        for f in valid:
            embed_text = _build_embed_text(f)
            texts.append(embed_text)
            self.file_paths.append(f["relative_path"])
            self.file_contents[f["relative_path"]] = f["full_text"]

        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # pre-normalize → dot product = cosine sim
        )

        self._save_cache()
        logger.info("Index build complete.")

    # ------------------------------------------------------------------
    # Incremental updates (called by watcher + agent after writes)
    # ------------------------------------------------------------------

    def update_file(self, relative_path: str, full_text: str) -> None:
        """Insert or update a single file in the index."""
        embed_text = full_text[:EMBED_CHARS]
        new_vec = self.model.encode(
            [embed_text], normalize_embeddings=True
        )[0]  # shape (dim,)

        if relative_path in self.file_paths:
            idx = self.file_paths.index(relative_path)
            self.embeddings[idx] = new_vec
        else:
            self.file_paths.append(relative_path)
            if self.embeddings is not None:
                self.embeddings = np.vstack(
                    [self.embeddings, new_vec.reshape(1, -1)]
                )
            else:
                self.embeddings = new_vec.reshape(1, -1)

        self.file_contents[relative_path] = full_text
        self._save_cache()

    def remove_file(self, relative_path: str) -> None:
        """Remove a file from the index."""
        if relative_path not in self.file_paths:
            return
        idx = self.file_paths.index(relative_path)
        self.file_paths.pop(idx)
        self.file_contents.pop(relative_path, None)
        if self.embeddings is not None:
            self.embeddings = np.delete(self.embeddings, idx, axis=0)
        self._save_cache()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Return the top-k most relevant (relative_path, score) pairs.

        Score is cosine similarity in [0, 1] because we pre-normalise.
        """
        if self.embeddings is None or len(self.file_paths) == 0:
            logger.warning("Index is empty — returning no results.")
            return []

        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores: np.ndarray = self.embeddings @ q_vec  # cosine similarity

        k = min(top_k, len(self.file_paths))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(self.file_paths[i], float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    def load_cache(self) -> bool:
        """Load from pickle cache.  Returns True if successful."""
        if not self.cache_path.exists():
            return False
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.file_paths = data["file_paths"]
            self.file_contents = data["file_contents"]
            logger.info(
                f"Loaded index cache: {len(self.file_paths)} files from {self.cache_path}"
            )
            return True
        except Exception as e:
            logger.warning(f"Cache load failed ({e}), will rebuild.")
            return False

    def _save_cache(self) -> None:
        with open(self.cache_path, "wb") as f:
            pickle.dump(
                {
                    "embeddings": self.embeddings,
                    "file_paths": self.file_paths,
                    "file_contents": self.file_contents,
                },
                f,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_content(self, relative_path: str) -> str:
        return self.file_contents.get(relative_path, "")

    @property
    def file_count(self) -> int:
        return len(self.file_paths)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_embed_text(file_dict: dict) -> str:
    """
    Build the string we actually embed for a file.
    Combines path (great signal for character/location names),
    frontmatter tags, and the first EMBED_CHARS of body text.
    """
    fm = file_dict.get("frontmatter", {}) or {}
    tags = fm.get("tags", [])
    tag_str = " ".join(tags) if isinstance(tags, list) else str(tags)
    name = fm.get("name", "")
    body = file_dict.get("content", "")[:EMBED_CHARS]
    return f"{file_dict['relative_path']} {name} {tag_str} {body}"
