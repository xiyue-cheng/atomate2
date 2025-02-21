"""Module defining functions for manipulating lobster files."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from atomate2.common.files import copy_files, get_zfile, gunzip_files
from atomate2.utils.file_client import FileClient, auto_fileclient
from atomate2.utils.path import strip_hostname
if TYPE_CHECKING:
    from pathlib import Path


ARTATOP_OUTPUT_FILES = ["re_lin", "re_nlin", "re_art"]

VASP_OUTPUT_FILES = [
    "INCAR",
    "vasprun.xml",
    "PROCAR",
    "WAVECAR",
    "OPTIC",
    "WAVEDER",
]

logger = logging.getLogger(__name__)


@auto_fileclient
def copy_artatop_files(
    src_dir: Path | str,
    src_host: str | None = None,
    file_client: FileClient = None,
) -> None:
    """
    Copy ARTATOP files to the current directory.

    This function will gunzip any gzipped files.

    Parameters
    ----------
    src_dir : Path or str
        The source directory.
    src_host : str or None
        The source hostname used to specify a remote filesystem. Can be given as
        either "username@remote_host" or just "remote_host" in which case the username
        will be inferred from the current user. If ``None``, the local filesystem will
        be used as the source.
    file_client : FileClient
        A file client to use for performing file operations.
    """
    
    src_dir = strip_hostname(src_dir)  # TODO: Handle hostnames properly.
    
    logger.info(f"Copying ARTATOP inputs from {src_dir}")
    directory_listing = file_client.listdir(src_dir, host=src_host)


    # Collect required files (VASP)
    files = []
    for file in VASP_OUTPUT_FILES:
        found_file = get_zfile(directory_listing, file, allow_missing=True)
        if found_file is not None:
            files.append(found_file)

    # Copy required files
    copy_files(
        src_dir,
        src_host=src_host,
        include_files=files,
        file_client=file_client,
    )

    # Decompress .gz files
    gunzip_files(
        include_files=files,
        allow_missing=True,
        file_client=file_client,
    )

    logger.info("Finished copying inputs")


def validate_required_files(required_files: list[str], directory: Path) -> None:
    """
    Validate that required files are present in a directory.

    Parameters
    ----------
    required_files : list of str
        List of required file names.
    directory : Path
        Directory to validate.

    Raises
    ------
    FileNotFoundError
        If any required files are missing.
    """
    missing_files = [file for file in required_files if not (directory / file).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing files: {', '.join(missing_files)} in {directory}"
        )
