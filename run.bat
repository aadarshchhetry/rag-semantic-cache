@echo off
set UV_CACHE_DIR=d:\aadarsh\AI\.uv_cache
set TEMP=d:\aadarsh\AI\.tmp
set TMP=d:\aadarsh\AI\.tmp
set HF_HOME=d:\aadarsh\AI\.hf_cache

echo Starting VeriCache with Qdrant Embedded & FastEmbed...
.\.venv\Scripts\python.exe run.py
pause
