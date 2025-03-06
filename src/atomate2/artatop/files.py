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
    dest_dir: Path | str,
    src_host: str | None = None,
    file_client: FileClient = None,
) -> None:
    """
    Copy ARTATOP files from the source directory to the destination directory.

    Parameters
    ----------
    src_dir : Path or str
        The source directory containing the input files.
    dest_dir : Path or str
        The destination directory where the files will be copied.
    src_host : str or None
        The source hostname for remote filesystems.
    file_client : FileClient
        A file client for performing file operations.
    """
    src_dir = strip_hostname(src_dir)
    dest_dir = strip_hostname(dest_dir)

    logger.info(f"Copying ARTATOP inputs from {src_dir} to {dest_dir}")
    directory_listing = file_client.listdir(src_dir, host=src_host)

    # Collect required files (e.g., WAVECAR, vasprun.xml)
    files = []
    for file in VASP_OUTPUT_FILES:  # Ensure VASP_OUTPUT_FILES is defined
        found_file = get_zfile(directory_listing, file, allow_missing=True)
        if found_file is not None:
            files.append(found_file)

    # Copy required files
    copy_files(
        src_dir,
        dest_dir,
        src_host=src_host,
        include_files=files,
        file_client=file_client,
    )

    # Decompress .gz files if necessary
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
