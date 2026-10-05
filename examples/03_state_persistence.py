#!/usr/bin/env python3
"""
03_state_persistence.py — Document Memory Freezing & Cross-Session Recall
========================================================================
Demonstrates serializing the Gist cognitive state of a document (in < 100 KB)
and loading it into a new session to query without re-reading the original document.
"""

import tempfile
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gist_memory import GistLayer, GistState


def main():
    print("=" * 65)
    print("  GIST MEMORY: COGNITIVE STATE PERSISTENCE")
    print("=" * 65)

    d_model = 512
    d_map = 32
    layer = GistLayer(d_model=d_model, d_map=d_map, decay=0.9999)
    layer.eval()

    # Step 1: Ingest document into associative memory
    doc_length = 5000
    print(f"[*] Ingesting long document ({doc_length} tokens) into Gist memory...")
    document_stream = torch.randn(1, doc_length, d_model)

    with torch.no_grad():
        _, doc_state = layer(document_stream, return_state=True)

    print(f"    -> Ingestion complete.")
    print(f"    -> Tokens consolidated: {doc_state.step_count}")
    print(f"    -> Gist state memory footprint: {doc_state.size_kb:.2f} KB")

    # Step 2: Save Gist snapshot to disk
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_file = Path(tmpdir) / "document_gist_snapshot.pt"
        print(f"\n[*] Freezing cognitive snapshot to: {snapshot_file.name}...")
        doc_state.save(snapshot_file)
        file_size_kb = snapshot_file.stat().st_size / 1024.0
        print(f"    -> Snapshot saved! File size on disk: {file_size_kb:.2f} KB")

        # Step 3: Simulate brand new session / process
        print(f"\n[*] Simulating new user query session without re-processing document...")
        restored_state = GistState.load(snapshot_file)

        # Step 4: Ask a question / query against the frozen document memory
        query_token = torch.randn(1, 1, d_model)
        with torch.no_grad():
            out, next_state = layer(query_token, state=restored_state, return_state=True)

        print(f"    -> Query successfully evaluated against restored Gist memory.")
        print(f"    -> Query output shape: {out.shape}")
        print(f"    -> Cumulative token horizon: {next_state.step_count}")

    print("\n[+] Done.")


if __name__ == "__main__":
    main()
