"""Render REPORT.md to REPORT.pdf (Markdown -> HTML -> headless Chrome/Edge print)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess

import markdown

CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; line-height: 1.38; color: #111; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 12.5pt; margin: 12pt 0 4pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 10.5pt; margin: 9pt 0 3pt; }
p, li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0; font-size: 8.4pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f4f4f4; padding: 0 2px; }
pre { background: #f4f4f4; padding: 5pt; font-size: 8.4pt; white-space: pre-wrap; }
img { max-width: 100%; display: block; margin: 4pt auto; page-break-inside: avoid; }
blockquote { margin: 4pt 0; padding-left: 8pt; border-left: 3px solid #ccc; color: #444; }
"""

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "msedge",
]


def find_browser():
    for b in BROWSERS:
        if os.path.exists(b) or shutil.which(b):
            return b
    raise SystemExit("no Chrome/Edge found for PDF printing")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="REPORT.md")
    p.add_argument("--output", default="REPORT.pdf")
    args = p.parse_args()

    root = os.path.dirname(os.path.abspath(args.input))
    with open(args.input, encoding="utf-8") as f:
        body = markdown.markdown(f.read(), extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    html_path = os.path.join(root, "REPORT.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    out = os.path.abspath(args.output)
    subprocess.run([
        find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={out}", "file:///" + html_path.replace("\\", "/"),
    ], check=True, capture_output=True)
    os.remove(html_path)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
