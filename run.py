import os
import sys
import subprocess

def main():
    print("=" * 70)
    print("  🚀 Starting VeriCache Control Plane")
    print("  - Vector Engine:   Qdrant Embedded (HNSW Graph Cosine)")
    print("  - Embedding Model: BAAI/bge-small-en-v1.5 (384-dim ONNX)")
    print("  - Verifier Model:  Xenova/ms-marco-MiniLM-L-6-v2 (Cross-Encoder)")
    print("  - Web Playground:  http://localhost:8000")
    print("  - Prometheus:      http://localhost:8000/metrics")
    print("=" * 70)

    # Set cache and temp directories to D drive
    os.environ["UV_CACHE_DIR"] = "d:/aadarsh/AI/.uv_cache"
    os.environ["TEMP"] = "d:/aadarsh/AI/.tmp"
    os.environ["TMP"] = "d:/aadarsh/AI/.tmp"
    os.environ["HF_HOME"] = "d:/aadarsh/AI/.hf_cache"

    python_executable = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe"))
    if not os.path.exists(python_executable):
        python_executable = sys.executable

    cmd = [
        python_executable,
        "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
