#!/usr/bin/env python3
"""
Entry point for Document Scanner & Summarizer CLI.

Usage:
    python main.py                              # Interactive mode
    python main.py document.pdf                 # Analyze specific file
    python main.py https://example.com/article  # Analyze web page
    python main.py document.pdf --summary-only  # Just get summary
"""

from src.cli import main

if __name__ == "__main__":
    main()
