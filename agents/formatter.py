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
    """Inline CSS optimised for xhtml2pdf PDF rendering with zero column overlap."""
    return """
        @page {
            size: A4 landscape;
            margin: 1.0cm 0.8cm 1.0cm 0.8cm;
        }
        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 8pt;
            line-height: 1.35;
            color: #1f2937;
        }
        h1 {
            font-size: 15pt;
            font-weight: bold;
            color: #1e1b4b;
            border-bottom: 2pt solid #4f46e5;
            padding-bottom: 4pt;
            margin-bottom: 8pt;
        }
        h2 {
            font-size: 12pt;
            font-weight: bold;
            color: #312e81;
            border-bottom: 1pt solid #e5e7eb;
            padding-bottom: 3pt;
            margin-top: 12pt;
            margin-bottom: 6pt;
        }
        p {
            margin-bottom: 6pt;
            text-align: justify;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            margin: 6pt 0;
            font-size: 7.5pt;
        }
        th {
            background: #4f46e5;
            color: white;
            padding: 5pt 4pt;
            text-align: left;
            font-weight: bold;
            font-size: 7.5pt;
            vertical-align: top;
            word-wrap: break-word;
        }
        td {
            padding: 4.5pt 4pt;
            border-bottom: 0.5pt solid #e5e7eb;
            font-size: 7.2pt;
            line-height: 1.35;
            vertical-align: top;
            word-wrap: break-word;
        }
        tr:nth-child(even) td {
            background: #f9fafb;
        }
        a {
            color: #4f46e5;
            text-decoration: underline;
            word-wrap: break-word;
        }
        hr {
            border: none;
            border-top: 1pt solid #e5e7eb;
            margin: 8pt 0;
        }
    """


def _get_web_css() -> str:
    """Inline CSS for the browser-rendered HTML report view."""
    css_path = Path(__file__).parent.parent / "utils" / "pdf_styles.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return """
        body { font-family: sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
        h1 { color: #1e1b4b; } h2 { color: #312e81; }
        table { width: 100%; border-collapse: collapse; table-layout: fixed; }
        th, td { padding: 8px; border: 1px solid #e5e7eb; word-break: break-word; }
        th { background: #4f46e5; color: white; }
    """


def _markdown_to_html(report_markdown: str, for_pdf: bool = False) -> str:
    """
    Convert Markdown to a complete HTML document.

    Args:
        report_markdown: The Markdown report text.
        for_pdf: If True, uses PDF-optimised inline CSS; otherwise uses web CSS.
    """
    import re
    extensions = ["tables", "fenced_code", "toc", "nl2br", "extra"]
    body_html = md_lib.markdown(report_markdown, extensions=extensions)

    if for_pdf:
        # Inject colgroup with strict widths for 7-column research table
        colgroup = """<colgroup>
            <col style="width: 22%;" width="22%" />
            <col style="width: 12%;" width="12%" />
            <col style="width: 6%;" width="6%" />
            <col style="width: 12%;" width="12%" />
            <col style="width: 16%;" width="16%" />
            <col style="width: 16%;" width="16%" />
            <col style="width: 16%;" width="16%" />
        </colgroup>"""
        body_html = re.sub(
            r"<table>",
            r'<table style="width: 100%; table-layout: fixed;">' + colgroup,
            body_html,
            count=1,
            flags=re.IGNORECASE,
        )

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

            meta_output = _PDF_DIR / f"{pdf_id}.json"
            try:
                import json
                meta_output.write_text(
                    json.dumps({"topic": state.get("user_query", "")}, ensure_ascii=False),
                    encoding="utf-8"
                )
            except Exception as e:
                print(f"[Formatter] ⚠ Could not save PDF metadata: {e}")
        else:
            print("[Formatter] PDF generation failed — HTML report still available")
    else:
        print("[Formatter] PDF skipped — xhtml2pdf not available")

    return {**state, "report_html": report_html, "pdf_path": pdf_path}


def get_pdf_dir() -> Path:
    """Return the directory where PDFs are stored (used by the API)."""
    return _PDF_DIR
