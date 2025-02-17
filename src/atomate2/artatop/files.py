"""Module defining functions for manipulating lobster files."""

from __future__ import annotations

import logging
from pathlib import Path

from atomate2.common.files import copy_files, get_zfile, gunzip_files
from atomate2.utils.file_client import FileClient, auto_fileclient
from atomate2.utils.path import strip_hostname

# Default file lists
VASP_OUTPUT_FILES = [
    "INCAR",
    "vasprun.xml",
    "PROCAR"
    "WAVECAR",
    "OPTIC",
    "WAVEDER",
]
ARTATOP_OUTPUT_FILES = ["re_lin", "re_nlin", "re_art"]
ARTATOP_OUTPUT_DIRS = ["out_lin", "out_nonlin"]

logger = logging.getLogger(__name__)


@auto_fileclient
def copy_artatop_files(
    src_dir: Path | str,
    src_host: str | None = None,
    file_client: FileClient = None,
) -> None:
    print(f"Original source directory: {src_dir}")  # Debugging
    """
    Copy VASP and ARTATOP files to the current directory.

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
    
     Force stripping hostname manually
    if ":" in str(src_dir):
        src_dir = str(src_dir).split(":", 1)[1]
    print(f"Stripped source directory: {src_dir}")  # Debugging

    # Ensure the path is absolute
    src_dir = Path(src_dir).resolve()
    print(f"Resolved source directory: {src_dir}")  # Debugging

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    directory_listing = file_client.listdir(src_dir, host=src_host)
    print(f"Files in source directory: {directory_listing}")  # Debugging

    # Collect required files (VASP and ARTATOP)
    files = []
    for file in VASP_OUTPUT_FILES + ARTATOP_OUTPUT_FILES:
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


def validate_required_dirs(required_dirs: list[str], directory: Path) -> bool:
    """
    Validate the presence of required directories in a directory.

    Parameters
    ----------
    required_dirs : list of str
        List of required directory names.
    directory : Path
        Directory to validate.

    Returns
    -------
    bool
        True if all directories are present, False otherwise.
    """
    directory = Path(directory)
    missing_dirs = [d for d in required_dirs if not (directory / d).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Missing directories: {', '.join(missing_dirs)} in {directory}"
        )
    return True


def validate_vasp_and_artatop_outputs(directory: Path) -> bool:
    """
    Validate all VASP and ARTATOP outputs together.

    Parameters
    ----------
    directory : Path
        Directory containing the output files.

    Returns
    -------
    bool
        True if all required files and directories are present, False otherwise.
    """
    logger.info(f"Validating files in {directory}")
    validate_required_files(VASP_OUTPUT_FILES + ARTATOP_OUTPUT_FILES, directory)
    validate_required_dirs(ARTATOP_OUTPUT_DIRS, directory)
    logger.info("Validation successful.")
    return True
