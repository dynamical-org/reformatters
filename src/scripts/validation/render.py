import re
from pathlib import Path

import typer
from markdown_it import MarkdownIt

from reformatters.common.logging import get_logger
from scripts.validation.utils import var_slug

log = get_logger(__name__)

REPORT_FILENAME = "validation_report.html"


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _extract_per_var_html(html: str) -> list[str]:
    # Per-variable headings render as <h3><code>name</code></h3>; other ### don't.
    return re.findall(r"<h3><code>([^<]+)</code></h3>", html)


_PLOT_TYPES = (
    ("availability", "availability over append dim"),
    ("value_timeseries", "full-period value time series"),
    ("spatial", "spatial comparison"),
    ("temporal", "time series comparison"),
)


def _wrap_variable_sections(html: str, available_files: set[str] | None) -> str:
    pattern = re.compile(
        r"<h3><code>(?P<var>[^<]+)</code></h3>(?P<body>.*?)(?=<h[23][\s>]|\Z)",
        re.DOTALL,
    )

    def replace(m: re.Match[str]) -> str:
        var = m.group("var")
        body = m.group("body")
        slug = var_slug(var)
        # Skip plots the run didn't produce (availability plots exist only for incomplete vars).
        plots = "".join(
            f'<a href="{filename}" target="_blank">'
            f'<img src="{filename}" alt="{var} — {label}"></a>'
            for prefix, label in _PLOT_TYPES
            if (filename := f"{prefix}_{slug}.png")
            and (available_files is None or filename in available_files)
        )
        return (
            f'<section class="variable" id="var-{slug}" data-var="{slug}">'
            f'<h3 class="var-heading"><code>{var}</code></h3>'
            f'{body}<div class="plots">{plots}</div></section>'
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


def _wrap_tables(html: str) -> str:
    return re.sub(
        r"(<table>.*?</table>)",
        r'<div class="table-scroll">\1</div>',
        html,
        flags=re.DOTALL,
    )


_CSS = """
:root {
  color-scheme: light dark;
  --bg-color: #ffffff;
  --text-color: #111111;
  --header-color: #111111;
  --link-color: #0b57d0;
  --link-visited-color: #6f42c1;
  --border-color: #111111;
  --border-muted-color: #444444;
  --muted-text: #666666;
  --muted-text-2: #999999;
  --pill-muted-bg: #f0f0f0;
  --pill-muted-fg: #111111;
  --pill-muted-border: #d0d0d0;
  --sidebar: 28rem;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg-color: #0f0f10;
    --text-color: #e8e8ea;
    --header-color: #ffffff;
    --link-color: #8ab4f8;
    --link-visited-color: #c58af9;
    --border-color: #e8e8ea;
    --border-muted-color: #b5b5b5;
    --muted-text: #b5b5b5;
    --muted-text-2: #8f8f93;
    --pill-muted-bg: #2a2a2d;
    --pill-muted-fg: #e8e8ea;
    --pill-muted-border: #3a3a3d;
  }
}

*, *::before, *::after { box-sizing: border-box; }
html { font-size: 62.5%; }
body, input, button, h1, h2, h3, h4, h5, h6 {
  font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}
body {
  margin: 0;
  font-size: 1.4rem;
  line-height: 1.6;
  color: var(--text-color);
  background-color: var(--bg-color);
}
a { color: var(--link-color); }
a:visited { color: var(--link-visited-color); }
h1, h2, h3, h4, h5, h6 {
  font-weight: 700;
  margin-top: 2rem;
  margin-bottom: 1rem;
  color: var(--header-color);
}
p { margin-bottom: 1.2rem; }

table { border-collapse: collapse; border: 1px solid var(--border-color);
        margin: 1rem 0 2rem; max-width: 100%; }
th, td { padding: 0.8rem 1.6rem; text-align: left; vertical-align: top;
         border-right: 1px dotted var(--border-muted-color); }
th { border-bottom: 1px solid var(--border-color); font-weight: 700; }
.table-scroll { overflow-x: auto; margin: 1rem 0 2rem; }
.table-scroll table { margin: 0; }

ul, ol { padding-left: 2rem; }
li { margin: 0.2rem 0; }

.toc-toggle {
  position: fixed; top: 1rem; left: 1rem; z-index: 30;
  border: 1px solid var(--border-color); background: var(--bg-color);
  color: var(--header-color);
  width: 3.6rem; height: 3.6rem; font-size: 1.6rem; cursor: pointer;
  display: none; padding: 0; line-height: 1;
  align-items: center; justify-content: center;
}
.toc-toggle:hover { background: var(--header-color); color: var(--bg-color); }

.toc {
  position: fixed; top: 0; left: 0; bottom: 0; width: var(--sidebar);
  border-right: 1px solid var(--border-color); padding: 2rem;
  overflow-y: auto; background: var(--bg-color); z-index: 20;
}
.toc-heading {
  font-size: 1.4rem; color: var(--header-color);
  margin: 2rem 0 0.8rem; font-weight: 700;
}
.toc-heading:first-child { margin-top: 0; }
.toc ul { list-style: none; padding: 0; margin: 0; }
.toc li { margin: 0.3rem 0; }
.toc a { color: var(--text-color); text-decoration: none; display: block; }
.toc a:visited { color: var(--text-color); }
.toc a:hover { color: var(--link-color); }
.toc .var-row { display: flex; align-items: center; gap: 0.6rem; }
.toc .var-row input { margin: 0; flex-shrink: 0; }
.toc .var-row a { flex: 1; min-width: 0; overflow: hidden;
                  text-overflow: ellipsis; white-space: nowrap;
                  font-size: 1.3rem; }
.toc .var-actions {
  display: flex; gap: 0.6rem; margin-bottom: 0.8rem;
  font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;
}
.toc .var-actions button {
  background: var(--bg-color); border: 1px solid var(--border-color);
  color: var(--header-color); padding: 0.2rem 0.8rem; cursor: pointer;
  font-weight: 700; letter-spacing: 1px;
}
.toc .var-actions button:hover { background: var(--header-color); color: var(--bg-color); }

main {
  margin-left: var(--sidebar); padding: 2rem 4rem 6rem;
  max-width: calc(78rem + var(--sidebar));
}
main h1 { margin-top: 0; }
main h2 { margin-top: 3.2rem; padding-bottom: 0.4rem;
          border-bottom: 1px solid var(--border-color); }
main h3 { margin-top: 2.4rem; }

section.variable {
  margin-top: 2.4rem; padding-top: 0.6rem;
  border-top: 1px solid var(--border-color);
}
section.variable.hidden { display: none; }
.plots { display: flex; flex-direction: column; gap: 1rem; margin: 1rem 0 2rem; }
.plots a { display: block; }
.plots img {
  display: block; max-width: 100%; height: auto;
  border: 1px solid var(--border-color);
  background: var(--bg-color);
}

@media (max-width: 880px) {
  .toc-toggle { display: flex; }
  .toc { transform: translateX(-100%); transition: transform 180ms ease;
         padding-top: 5.6rem;
         box-shadow: 0.4rem 0 1.2rem var(--shadow-color, rgba(0,0,0,0.4)); }
  body.toc-open .toc { transform: translateX(0); }
  body.toc-open::after { content: ""; position: fixed; inset: 0;
                         background: rgba(0,0,0,0.4); z-index: 15; }
  main { margin-left: 0; padding: 6rem 2rem 3rem; }
  table { font-size: 1.2rem; }
  th, td { padding: 0.4rem 0.8rem; }
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
    sections: list[tuple[str, str]], variables: list[str], dataset_name: str
) -> str:
    section_items = "".join(
        f'<li><a href="#{slug}">{title}</a></li>' for slug, title in sections
    )
    var_items = "".join(
        f'<li class="var-row"><input type="checkbox" checked data-var="{var_slug(v)}" '
        f'id="cb-{var_slug(v)}"><a href="#var-{var_slug(v)}">{v}</a></li>'
        for v in variables
    )
    return f"""
<nav class="toc" aria-label="Table of contents">
  <div class="toc-heading">{dataset_name}</div>
  <ul>{section_items}</ul>
  <div class="toc-heading">Variables</div>
  <div class="var-actions">
    <button type="button" data-action="all">all</button>
    <button type="button" data-action="none">none</button>
  </div>
  <ul>{var_items}</ul>
</nav>
"""


def _extract_dataset_name(md_text: str, fallback: str) -> str:
    m = re.search(r"^\|\s*Validation\s*\|\s*([^|]+?)\s*\|", md_text, flags=re.MULTILINE)
    return m.group(1) if m else fallback


def render_html(
    md_text: str, dataset_id: str, available_files: set[str] | None = None
) -> str:
    md = MarkdownIt("commonmark").enable("table")
    html = md.render(md_text)
    html = _png_links_open_in_new_tab(html)
    variables = _extract_per_var_html(html)
    html, sections = _annotate_h2(html)
    html = _annotate_non_var_h3(html)
    html = _wrap_variable_sections(html, available_files)
    html = _wrap_tables(html)
    dataset_name = _extract_dataset_name(md_text, dataset_id)
    toc = _build_toc(sections, variables, dataset_name)
    title = f"Validation report — {dataset_id}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap" rel="stylesheet">
<link rel="icon" href="https://assets.dynamical.org/identity/logo/favicon/favicon.svg" type="image/svg+xml">
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
    available_files = {f.name for f in run_dir.iterdir() if f.is_file()}
    out_path.write_text(render_html(md_path.read_text(), dataset_id, available_files))
    return out_path


run_dir_argument = typer.Argument(..., help="Path to the run directory")


def render_report_command(run_dir: Path = run_dir_argument) -> None:
    out = render_report(run_dir)
    log.info(f"Rendered: {out}")
    typer.echo(str(out))
