"""
watcher.py — Monitors the vault directory for changes made outside of
this application (e.g. you edit a file directly in Obsidian).

Uses watchdog's PollingObserver instead of the default inotify-based
Observer because inotify does NOT work over SMB/NFS network shares.
PollingObserver polls the filesystem on a timer — it's slightly less
instant than inotify but is reliable across all mount types.

Tune POLL_INTERVAL_SECONDS to balance responsiveness vs NAS load.
"""

import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from vault import VaultManager
from embeddings import VaultIndex

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 30   # How often to check for changes on the NAS


class _VaultEventHandler(FileSystemEventHandler):
    def __init__(self, vault: VaultManager, index: VaultIndex):
        super().__init__()
        self.vault = vault
        self.index = index
        self._lock = threading.Lock()

    # ---- helpers ----

    def _relative(self, abs_path: str) -> str | None:
        try:
            rel = str(Path(abs_path).relative_to(self.vault.vault_path))
            # Skip hidden files / obsidian internals
            if any(part.startswith(".") for part in Path(rel).parts):
                return None
            return rel
        except ValueError:
            return None

    # ---- event handlers ----

    def on_modified(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(".md"):
            return
        rel = self._relative(event.src_path)
        if not rel:
            return
        with self._lock:
            try:
                full_text = Path(event.src_path).read_text(encoding="utf-8")
                self.index.update_file(rel, full_text)
                logger.info(f"[watcher] re-indexed modified: {rel}")
            except Exception as e:
                logger.warning(f"[watcher] could not re-index {rel}: {e}")

    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(".md"):
            return
        rel = self._relative(event.src_path)
        if not rel:
            return
        with self._lock:
            try:
                full_text = Path(event.src_path).read_text(encoding="utf-8")
                self.index.update_file(rel, full_text)
                logger.info(f"[watcher] indexed new file: {rel}")
            except Exception as e:
                logger.warning(f"[watcher] could not index {rel}: {e}")

    def on_deleted(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(".md"):
            return
        rel = self._relative(event.src_path)
        if not rel:
            return
        with self._lock:
            self.index.remove_file(rel)
            logger.info(f"[watcher] removed from index: {rel}")

    def on_moved(self, event):
        """Handle renames / moves."""
        if event.is_directory:
            return

        src_rel = self._relative(event.src_path)
        dest_rel = self._relative(event.dest_path)

        with self._lock:
            if src_rel and src_rel.endswith(".md"):
                self.index.remove_file(src_rel)

            if dest_rel and dest_rel.endswith(".md"):
                try:
                    full_text = Path(event.dest_path).read_text(encoding="utf-8")
                    self.index.update_file(dest_rel, full_text)
                    logger.info(f"[watcher] re-indexed moved: {src_rel} → {dest_rel}")
                except Exception as e:
                    logger.warning(f"[watcher] could not index {dest_rel}: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_watcher(vault: VaultManager, index: VaultIndex) -> PollingObserver:
    """
    Start a background PollingObserver that keeps the index in sync with
    any external changes to the vault (e.g. direct Obsidian edits).

    Returns the observer so the caller can stop it on shutdown.
    """
    handler = _VaultEventHandler(vault, index)
    observer = PollingObserver(timeout=POLL_INTERVAL_SECONDS)
    observer.schedule(handler, str(vault.vault_path), recursive=True)
    observer.start()
    logger.info(
        f"[watcher] Polling vault every {POLL_INTERVAL_SECONDS}s: {vault.vault_path}"
    )
    return observer
