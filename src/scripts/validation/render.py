import re
from pathlib import Path

import typer
from markdown_it import MarkdownIt

from reformatters.common.logging import get_logger

log = get_logger(__name__)

REPORT_FILENAME = "validation_report.html"


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _extract_per_var_html(html: str) -> list[str]:
    # Per-variable headings render as <h3><code>name</code></h3>; other ### don't.
    return re.findall(r"<h3><code>([^<]+)</code></h3>", html)


def _wrap_variable_sections(html: str) -> str:
    pattern = re.compile(
        r"<h3><code>(?P<var>[^<]+)</code></h3>(?P<body>.*?)(?=<h[23][\s>]|\Z)",
        re.DOTALL,
    )

    def replace(m: re.Match[str]) -> str:
        var = m.group("var")
        body = m.group("body")
        plots = (
            '<div class="plots">'
            f'<a href="nulls_{var}.png" target="_blank">'
            f'<img src="nulls_{var}.png" alt="{var} — null fraction"></a>'
            f'<a href="spatial_{var}.png" target="_blank">'
            f'<img src="spatial_{var}.png" alt="{var} — spatial comparison"></a>'
            f'<a href="temporal_{var}.png" target="_blank">'
            f'<img src="temporal_{var}.png" alt="{var} — time series comparison"></a>'
            "</div>"
        )
        return (
            f'<section class="variable" id="var-{var}" data-var="{var}">'
            f'<h3 class="var-heading"><code>{var}</code></h3>'
            f"{body}{plots}</section>"
        )

    return pattern.sub(replace, html)


def _annotate_h2(html: str) -> tuple[str, list[tuple[str, str]]]:
    sections: list[tuple[str, str]] = []

    def replace(m: re.Match[str]) -> str:
        inner = m.group(1)
        plain = re.sub(r"<[^>]+>", "", inner).strip()
        slug = _slugify(plain)
        sections.append((slug, plain))
        return f'<h2 id="{slug}">{inner}</h2>'

    out = re.sub(r"<h2>(.*?)</h2>", replace, html, flags=re.DOTALL)
    return out, sections


def _annotate_non_var_h3(html: str) -> str:
    def replace(m: re.Match[str]) -> str:
        inner = m.group(1)
        if inner.startswith("<code>"):
            return m.group(0)
        plain = re.sub(r"<[^>]+>", "", inner).strip()
        slug = _slugify(plain)
        return f'<h3 id="{slug}">{inner}</h3>'

    return re.sub(r"<h3>(.*?)</h3>", replace, html, flags=re.DOTALL)


def _png_links_open_in_new_tab(html: str) -> str:
    # Markdown-rendered links to .png files: open in a new tab so users can zoom.
    return re.sub(
        r'<a href="([^"]+\.png)">',
        r'<a href="\1" target="_blank">',
        html,
    )


_CSS = """
:root { --sidebar: 280px; --border: #e2e2e2; --bg: #fff; --text: #222;
        --muted: #666; --accent: #1f6feb; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
       Helvetica, Arial, sans-serif; color: var(--text); background: var(--bg); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
       font-size: 0.92em; background: #f4f4f4; padding: 0.1em 0.3em;
       border-radius: 3px; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; margin: 0.5rem 0 1rem; max-width: 100%;
        display: block; overflow-x: auto; }
th, td { border: 1px solid var(--border); padding: 0.4rem 0.6rem;
         text-align: left; vertical-align: top; }
th { background: #fafafa; }

.toc-toggle { position: fixed; top: 0.6rem; left: 0.6rem; z-index: 30;
              border: 1px solid var(--border); background: #fff;
              border-radius: 6px; width: 38px; height: 38px;
              font-size: 1.2rem; cursor: pointer; display: none; }
.toc { position: fixed; top: 0; left: 0; bottom: 0; width: var(--sidebar);
       border-right: 1px solid var(--border); padding: 1rem 1rem 2rem;
       overflow-y: auto; background: var(--bg); z-index: 20; }
.toc h2 { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
          color: var(--muted); margin: 1rem 0 0.4rem; font-weight: 600; }
.toc h2:first-child { margin-top: 0; }
.toc ul { list-style: none; padding: 0; margin: 0; }
.toc li { margin: 0.15rem 0; }
.toc a { color: var(--text); display: block; padding: 0.15rem 0; }
.toc a:hover { color: var(--accent); }
.toc .var-row { display: flex; align-items: center; gap: 0.4rem; }
.toc .var-row input { margin: 0; }
.toc .var-row a { padding: 0; flex: 1; min-width: 0; overflow: hidden;
                  text-overflow: ellipsis; white-space: nowrap; }
.toc .var-actions { display: flex; gap: 0.4rem; margin-bottom: 0.4rem;
                    font-size: 0.8rem; }
.toc .var-actions button { background: none; border: 1px solid var(--border);
                           padding: 0.1rem 0.4rem; cursor: pointer;
                           border-radius: 3px; color: var(--muted); }
.toc .var-actions button:hover { color: var(--accent); border-color: var(--accent); }

main { margin-left: var(--sidebar); padding: 1.5rem 2rem 4rem; max-width: 70rem; }
main h1 { margin-top: 0; }
main h2 { margin-top: 2.2rem; padding-bottom: 0.3rem;
          border-bottom: 1px solid var(--border); }
main h3 { margin-top: 1.6rem; }

section.variable { margin-top: 1.6rem; padding-top: 0.4rem;
                   border-top: 1px solid var(--border); }
section.variable.hidden { display: none; }
.plots { display: flex; flex-direction: column; gap: 0.6rem; margin: 0.6rem 0; }
.plots a { display: block; }
.plots img { display: block; max-width: 100%; height: auto;
             border: 1px solid var(--border); border-radius: 4px; }

@media (max-width: 880px) {
  .toc-toggle { display: block; }
  .toc { transform: translateX(-100%); transition: transform 180ms ease;
         box-shadow: 2px 0 12px rgba(0,0,0,0.08); }
  body.toc-open .toc { transform: translateX(0); }
  body.toc-open::after { content: ""; position: fixed; inset: 0;
                         background: rgba(0,0,0,0.3); z-index: 15; }
  main { margin-left: 0; padding: 3.5rem 1rem 3rem; }
}
"""


_JS = r"""
(function () {
  var body = document.body;
  var toggle = document.querySelector('.toc-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      body.classList.toggle('toc-open');
    });
  }
  document.querySelectorAll('.toc a').forEach(function (a) {
    a.addEventListener('click', function () {
      body.classList.remove('toc-open');
      var href = a.getAttribute('href') || '';
      if (href.indexOf('#var-') === 0) {
        var v = href.slice(5);
        var cb = document.querySelector('input[data-var="' + cssEscape(v) + '"]');
        if (cb && !cb.checked) { cb.checked = true; applyVar(v, true); }
      }
    });
  });

  function cssEscape(s) {
    return s.replace(/(["\\])/g, '\\$1');
  }
  function applyVar(v, on) {
    document.querySelectorAll('section.variable[data-var="' + cssEscape(v) + '"]')
      .forEach(function (s) { s.classList.toggle('hidden', !on); });
  }
  document.querySelectorAll('input[data-var]').forEach(function (cb) {
    cb.addEventListener('change', function () { applyVar(cb.dataset.var, cb.checked); });
  });
  var allBtn = document.querySelector('[data-action="all"]');
  var noneBtn = document.querySelector('[data-action="none"]');
  if (allBtn) allBtn.addEventListener('click', function () { setAll(true); });
  if (noneBtn) noneBtn.addEventListener('click', function () { setAll(false); });
  function setAll(on) {
    document.querySelectorAll('input[data-var]').forEach(function (cb) {
      cb.checked = on; applyVar(cb.dataset.var, on);
    });
  }
})();
"""


def _build_toc(
    sections: list[tuple[str, str]], variables: list[str], dataset_id: str
) -> str:
    section_items = "".join(
        f'<li><a href="#{slug}">{title}</a></li>' for slug, title in sections
    )
    var_items = "".join(
        f'<li class="var-row"><input type="checkbox" checked data-var="{v}" '
        f'id="cb-{v}"><a href="#var-{v}">{v}</a></li>'
        for v in variables
    )
    return f"""
<nav class="toc" aria-label="Table of contents">
  <h2>{dataset_id}</h2>
  <ul>{section_items}</ul>
  <h2>Variables</h2>
  <div class="var-actions">
    <button type="button" data-action="all">all</button>
    <button type="button" data-action="none">none</button>
  </div>
  <ul>{var_items}</ul>
</nav>
"""


def render_html(md_text: str, dataset_id: str) -> str:
    md = MarkdownIt("commonmark").enable("table")
    html = md.render(md_text)
    html = _png_links_open_in_new_tab(html)
    variables = _extract_per_var_html(html)
    html, sections = _annotate_h2(html)
    html = _annotate_non_var_h3(html)
    html = _wrap_variable_sections(html)
    toc = _build_toc(sections, variables, dataset_id)
    title = f"Validation report — {dataset_id}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<button type="button" class="toc-toggle" aria-label="Open table of contents">☰</button>
{toc}
<main>{html}</main>
<script>{_JS}</script>
</body>
</html>
"""


def render_report(run_dir: Path) -> Path:
    md_path = run_dir / "validation_summary.md"
    assert md_path.exists(), f"validation_summary.md not found in {run_dir}"
    dataset_id = run_dir.parent.name
    out_path = run_dir / REPORT_FILENAME
    out_path.write_text(render_html(md_path.read_text(), dataset_id))
    return out_path


run_dir_argument = typer.Argument(..., help="Path to the run directory")


def render_report_command(run_dir: Path = run_dir_argument) -> None:
    out = render_report(run_dir)
    log.info(f"Rendered: {out}")
    typer.echo(str(out))
