import pandas as pd
import typer

from noaa.gefs.forecast import reformat, template

app = typer.Typer()


@app.command()
def update_template() -> None:
    template.update_template()


@app.command()
def reformat_local(init_time_end: str) -> None:
    reformat.reformat_local(init_time_end)


@app.command()
def reformat_kubernetes(
    init_time_end: str, jobs_per_pod: int = 10, max_parallelism: int = 32
) -> None:
    reformat.reformat_kubernetes(init_time_end, jobs_per_pod, max_parallelism)


@app.command()
def reformat_chunks(init_time_end: str, worker_index: int, workers_total: int) -> None:
    reformat.reformat_chunks(
        init_time_end, worker_index=worker_index, workers_total=workers_total
    )


if __name__ == "__main__":
    app()
