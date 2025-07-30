from pathlib import Path

import icechunk


def get_writable_session(
    path: Path | str,
    branch_name: str = "main",
) -> icechunk.session.Session:
    """Get/create icechunk store parallel to regular store"""
    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session(branch_name)
    return session
