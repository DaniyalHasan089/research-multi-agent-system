"""
Agent 5: Formatter (Pure Python — no LLM)

Converts the Markdown report into:
  1. Styled HTML (inline CSS, no external dependencies)
  2. A PDF file via xhtml2pdf (pure Python, no system libraries required)

Output is written to ResearchState: report_html, pdf_path

This agent is fully deterministic — no reasoning required.
PDF generation degrades gracefully: if xhtml2pdf is unavailable,
report_html is still produced and the pipeline completes normally.
"""

import os
import tempfile
import uuid
from pathlib import Path
from io import BytesIO

import markdown as md_lib
from graph.state import ResearchState


# Directory to store generated PDFs (temp, cleaned up on server restart)
_PDF_DIR = Path(tempfile.gettempdir()) / "research_agent_pdfs"
_PDF_DIR.mkdir(parents=True, exist_ok=True)

# Try to import xhtml2pdf — fail gracefully if not installed
try:
    from xhtml2pdf import pisa
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False
    print("[Formatter] ⚠ xhtml2pdf not installed — PDF export disabled. "
          "Run: pip install xhtml2pdf")


def _get_pdf_css() -> str:
    """Inline CSS optimised for xhtml2pdf PDF rendering."""
    return """
        @page {
            size: A4;
            margin: 2cm 1.8cm 2.5cm 1.8cm;
        }
        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #1f2937;
        }
        h1 {
            font-size: 20pt;
            font-weight: bold;
            color: #1e1b4b;
            border-bottom: 2pt solid #4f46e5;
            padding-bottom: 6pt;
            margin-bottom: 12pt;
        }
        h2 {
            font-size: 14pt;
            font-weight: bold;
            color: #312e81;
            border-bottom: 1pt solid #e5e7eb;
            padding-bottom: 4pt;
            margin-top: 18pt;
            margin-bottom: 8pt;
        }
        h3 {
            font-size: 12pt;
            font-weight: bold;
            color: #4338ca;
            margin-top: 12pt;
            margin-bottom: 6pt;
        }
        p {
            margin-bottom: 8pt;
            text-align: justify;
        }
        ul, ol {
            margin-left: 18pt;
            margin-bottom: 8pt;
        }
        li {
            margin-bottom: 3pt;
        }
        a {
            color: #4f46e5;
        }
        code {
            font-family: Courier, monospace;
            font-size: 9pt;
            background: #f3f4f6;
            padding: 1pt 3pt;
        }
        pre {
            background: #f8f9fa;
            border-left: 3pt solid #4f46e5;
            padding: 8pt 10pt;
            font-size: 9pt;
            font-family: Courier, monospace;
            margin-bottom: 10pt;
        }
        blockquote {
            border-left: 3pt solid #6366f1;
            padding: 4pt 10pt;
            background: #eef2ff;
            margin: 8pt 0;
            font-style: italic;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10pt 0;
            font-size: 10pt;
        }
        th {
            background: #4f46e5;
            color: white;
            padding: 5pt 8pt;
            text-align: left;
            font-weight: bold;
        }
        td {
            padding: 4pt 8pt;
            border-bottom: 1pt solid #e5e7eb;
        }
        tr:nth-child(even) td {
            background: #f9fafb;
        }
        hr {
            border: none;
            border-top: 1pt solid #e5e7eb;
            margin: 12pt 0;
        }
    """


def _get_web_css() -> str:
    """Inline CSS for the browser-rendered HTML report view."""
    css_path = Path(__file__).parent.parent / "utils" / "pdf_styles.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    # Minimal fallback
    return """
        body { font-family: sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; }
        h1 { color: #1e1b4b; } h2 { color: #312e81; }
        a { color: #4f46e5; } code { background: #f3f4f6; padding: 2px 6px; border-radius: 3px; }
    """


def _markdown_to_html(report_markdown: str, for_pdf: bool = False) -> str:
    """
    Convert Markdown to a complete HTML document.

    Args:
        report_markdown: The Markdown report text.
        for_pdf: If True, uses PDF-optimised inline CSS; otherwise uses web CSS.
    """
    extensions = ["tables", "fenced_code", "toc", "nl2br", "extra"]
    body_html = md_lib.markdown(report_markdown, extensions=extensions)
    css = _get_pdf_css() if for_pdf else _get_web_css()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Research Report</title>
    <style>{css}</style>
</head>
<body>
{body_html}
</body>
</html>"""


def _generate_pdf(html_content: str, output_path: str) -> bool:
    """
    Render HTML to PDF using xhtml2pdf.

    Returns True on success, False on failure.
    """
    if not _PDF_AVAILABLE:
        return False

    try:
        with open(output_path, "wb") as f:
            result = pisa.CreatePDF(
                src=html_content,
                dest=f,
                encoding="utf-8",
            )
        if result.err:
            print(f"[Formatter] ⚠ xhtml2pdf reported errors: {result.err}")
            return False
        return True
    except Exception as e:
        print(f"[Formatter] ⚠ PDF generation failed: {e}")
        return False


def run(state: ResearchState) -> ResearchState:
    """
    Formatter node — converts Markdown → HTML (for browser) + PDF.

    Args:
        state: Current ResearchState with 'report_markdown'.

    Returns:
        Updated ResearchState with 'report_html' and 'pdf_path'.
    """
    report_markdown = state["report_markdown"]

    # ── Generate web HTML (with full dark-mode CSS for browser view) ──────────
    print("[Formatter] Converting Markdown → HTML")
    report_html = _markdown_to_html(report_markdown, for_pdf=False)
    print(f"[Formatter] HTML generated ({len(report_html)} chars)")

    # ── Generate PDF ──────────────────────────────────────────────────────────
    pdf_path = ""
    if _PDF_AVAILABLE:
        pdf_id = str(uuid.uuid4())
        pdf_output = str(_PDF_DIR / f"{pdf_id}.pdf")

        # Render a separate PDF-optimised HTML (lighter CSS for xhtml2pdf)
        pdf_html = _markdown_to_html(report_markdown, for_pdf=True)

        print(f"[Formatter] Rendering PDF → {pdf_output}")
        success = _generate_pdf(pdf_html, pdf_output)

        if success:
            size_kb = os.path.getsize(pdf_output) / 1024
            print(f"[Formatter] ✓ PDF saved ({size_kb:.1f} KB)")
            pdf_path = pdf_output
        else:
            print("[Formatter] PDF generation failed — HTML report still available")
    else:
        print("[Formatter] PDF skipped — xhtml2pdf not available")

    return {**state, "report_html": report_html, "pdf_path": pdf_path}


def get_pdf_dir() -> Path:
    """Return the directory where PDFs are stored (used by the API)."""
    return _PDF_DIR
