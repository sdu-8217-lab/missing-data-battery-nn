#!/usr/bin/env python3
"""Simple runner for analysis with limited output"""
import subprocess
result = subprocess.run(['python', 'analyze_search_results.py'], capture_output=True, text=True)
print(result.stdout[:8000])  # First 8000 chars
if result.stderr:
    print("ERRORS:", result.stderr[:1000])
