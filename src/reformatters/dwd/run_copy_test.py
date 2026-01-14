import sys
from pathlib import PurePosixPath

import typer

from reformatters.dwd.copy_files_from_dwd_ftp import copy_files_from_dwd_ftp

app = typer.Typer()


@app.command()
def main(
    dst_root: str = typer.Argument(..., help="The local destination root directory."),
    nwp_init_hour: int = typer.Option(
        0,
        help="The NWP initialisation hour, e.g. 0. This will be used as the FTP path in this form: '/weather/nwp/icon-eu/grib/{nwp_init_hour:02d}'",
    ),
    ftp_host: str = typer.Option(
        "opendata.dwd.de", help="The FTP host, e.g. 'opendata.dwd.de'"
    ),
    transfers: int = typer.Option(10, help="Number of parallel transfers."),
    max_files: int = typer.Option(
        sys.maxsize, help="Max number of files to transfer per NWP variable."
    ),
) -> None:
    """
    Simple script to test copying and restructuring files from DWD's FTP server.
    """
    copy_files_from_dwd_ftp(
        ftp_host=ftp_host,
        ftp_path=PurePosixPath("/weather/nwp/icon-eu/grib") / f"{nwp_init_hour:02d}",
        dst_root=PurePosixPath(dst_root),
        transfers=transfers,
        max_files_per_nwp_variable=max_files,
    )


if __name__ == "__main__":
    app()
