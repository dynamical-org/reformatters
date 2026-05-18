from pathlib import Path

from scripts.validation.render import REPORT_FILENAME, render_html, render_report

FIXTURE_MD = """\
# Validation run — `noaa-test` `v9.9.9`

## Datasets

| Role | Name |
|---|---|
| Validation | NOAA Test |

## Summary

### For further review

- spatial map looks fine

### What looks good

- everything else

## Combined plots

- nulls: [`combined_nulls.png`](combined_nulls.png)

## Per-variable details

### `temperature_2m`

**Metadata**

- units: `degree_Celsius`

### `precipitation_surface`

**Metadata**

- units: `kg m-2 s-1`
"""


def test_render_html_structure() -> None:
    html = render_html(FIXTURE_MD, "noaa-test")

    assert '<h2 id="datasets">Datasets</h2>' in html
    assert '<h2 id="summary">Summary</h2>' in html
    assert '<h2 id="combined-plots">Combined plots</h2>' in html
    assert '<h2 id="per-variable-details">Per-variable details</h2>' in html

    # Non-variable h3s get ids but are NOT wrapped in <section>.
    assert '<h3 id="for-further-review">For further review</h3>' in html
    assert '<h3 id="what-looks-good">What looks good</h3>' in html

    # Variable h3s are wrapped in <section> with id="var-<name>".
    assert (
        '<section class="variable" id="var-temperature_2m" data-var="temperature_2m">'
        in html
    )
    assert (
        '<section class="variable" id="var-precipitation_surface" '
        'data-var="precipitation_surface">'
    ) in html

    # Each variable section embeds its three plot images, each wrapped in <a target=_blank>.
    for plot_type in ("nulls", "spatial", "temporal"):
        assert (
            f'<a href="{plot_type}_temperature_2m.png" target="_blank">'
            f'<img src="{plot_type}_temperature_2m.png"'
        ) in html

    # PNG links from elsewhere in the doc get target="_blank" too.
    assert '<a href="combined_nulls.png" target="_blank">' in html

    # TOC contains a checkbox per variable + section links.
    assert 'data-var="temperature_2m"' in html
    assert 'data-var="precipitation_surface"' in html
    assert '<a href="#for-further-review">' not in html  # h3s not in top-level TOC
    assert '<a href="#datasets">Datasets</a>' in html

    assert "<head>" in html
    assert "<title>" in html


def test_render_report_writes_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "noaa-test" / "v9.9.9_2026-01-01T00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "validation_summary.md").write_text(FIXTURE_MD)
    out = render_report(run_dir)
    assert out == run_dir / REPORT_FILENAME
    assert out.exists()
    body = out.read_text()
    assert "<!doctype html>" in body
    assert "var-temperature_2m" in body
