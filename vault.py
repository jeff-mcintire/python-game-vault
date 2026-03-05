"""
vault.py — Manages all read/write operations on the Obsidian vault.

Treats the vault as a plain filesystem path, which works for both local
mounts and NAS shares (SMB/NFS mounted at a known path).
"""

import shutil
import logging
from pathlib import Path
from typing import Optional

import frontmatter

logger = logging.getLogger(__name__)


class VaultManager:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path).resolve()
        if not self.vault_path.exists():
            raise FileNotFoundError(
                f"Vault path not found: {vault_path}\n"
                "If this is a NAS share, make sure it is mounted before starting the server."
            )
        logger.info(f"Vault initialized at: {self.vault_path}")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def scan_files(self) -> list[Path]:
        """Recursively return all .md files, skipping hidden dirs (.obsidian, .trash)."""
        files = []
        for p in self.vault_path.rglob("*.md"):
            # Skip Obsidian internals and trash
            if any(part.startswith(".") for part in p.parts):
                continue
            files.append(p)
        return files

    def read_file(self, path: Path) -> Optional[dict]:
        """
        Read a markdown file and return a dict with:
          - relative_path  (str)
          - content        (body without frontmatter)
          - frontmatter    (dict)
          - full_text      (raw file text — used for embedding)
        """
        try:
            raw = path.read_text(encoding="utf-8")
            post = frontmatter.loads(raw)
            return {
                "path": str(path),
                "relative_path": str(path.relative_to(self.vault_path)),
                "content": post.content,
                "frontmatter": dict(post.metadata),
                "full_text": raw,
            }
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            return None

    def read_relative(self, relative_path: str) -> str:
        """Return full raw text of a file by its vault-relative path."""
        target = self.vault_path / relative_path
        if not target.exists():
            raise FileNotFoundError(f"File not found in vault: {relative_path}")
        return target.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write_file(
        self,
        relative_path: str,
        content: str,
        frontmatter_data: Optional[dict] = None,
    ) -> str:
        """
        Create or fully overwrite a file.
        Returns the canonical relative path string.
        """
        target = self.vault_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)

        if frontmatter_data:
            post = frontmatter.Post(content, **frontmatter_data)
            target.write_text(frontmatter.dumps(post), encoding="utf-8")
        else:
            target.write_text(content, encoding="utf-8")

        rel = str(target.relative_to(self.vault_path))
        logger.info(f"Wrote: {rel}")
        return rel

    def append_file(self, relative_path: str, content: str) -> str:
        """Append a markdown block to an existing file."""
        target = self.vault_path / relative_path
        if not target.exists():
            raise FileNotFoundError(f"Cannot append — file not found: {relative_path}")
        with open(target, "a", encoding="utf-8") as f:
            f.write(f"\n\n{content}")
        rel = str(target.relative_to(self.vault_path))
        logger.info(f"Appended to: {rel}")
        return rel

    def delete_file(self, relative_path: str) -> str:
        """
        Soft-delete: moves the file to a .trash folder inside the vault
        rather than permanently deleting it.
        """
        target = self.vault_path / relative_path
        if not target.exists():
            raise FileNotFoundError(f"Cannot delete — file not found: {relative_path}")
        trash = self.vault_path / ".trash"
        trash.mkdir(exist_ok=True)
        dest = trash / target.name
        shutil.move(str(target), str(dest))
        logger.info(f"Moved to trash: {relative_path}")
        return relative_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_structure(self) -> str:
        """Return a compact tree of all vault .md files for context."""
        lines = []
        for path in sorted(self.scan_files()):
            rel = path.relative_to(self.vault_path)
            lines.append(str(rel))
        return "\n".join(lines)
