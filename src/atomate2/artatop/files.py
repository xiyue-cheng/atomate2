import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default file lists
DEFAULT_VASP_OUTPUT_FILES = [
    "OUTCAR",
    "vasprun.xml",
    "CHG",
    "CHGCAR",
    "CONTCAR",
    "INCAR",
    "KPOINTS",
    "POSCAR",
    "POTCAR",
    "DOSCAR",
    "EIGENVAL",
    "IBZKPT",
    "OSZICAR",
    "WAVECAR",
    "XDATCAR",
    "OPTIC",
]
DEFAULT_RELAXATION_FILES = ["OUTCAR", "CONTCAR"]
DEFAULT_OPTICS_FILES = ["vasprun.xml"]
DEFAULT_ARTATOP_OUTPUT_FILES = ["re_lin", "re_nlin", "re_art"]
DEFAULT_ARTATOP_OUTPUT_DIRS = ["out_lin", "out_nonlin"]

# ---------------------------------------------------------
# FILE VALIDATION FUNCTIONS
# ---------------------------------------------------------


def validate_required_files(required_files, directory):
    """
    Validate that required files are present in a directory.

    Parameters
    ----------
    required_files : list of str
        List of required file names.
    directory : Path
        Path to the directory to check.

    Returns
    -------
    bool
        True if all required files are present, False otherwise.
    """
    directory = Path(directory)
    missing_files = [file for file in required_files if not (directory / file).exists()]
    if missing_files:
        logger.error(
            f"Missing required files: {', '.join(missing_files)} in {directory}"
        )
        return False
    return True


def validate_required_dirs(required_dirs, base_directory):
    """
    Validate that required directories are present.

    Parameters
    ----------
    required_dirs : list of str
        List of required directory names.
    base_directory : Path
        Path to the base directory to check.

    Returns
    -------
    bool
        True if all required directories are present, False otherwise.
    """
    base_directory = Path(base_directory)
    missing_dirs = [
        dir_name
        for dir_name in required_dirs
        if not (base_directory / dir_name).is_dir()
    ]
    if missing_dirs:
        logger.error(
            f"Missing required directories: {', '.join(missing_dirs)} in {base_directory}"
        )
        return False
    return True


# ---------------------------------------------------------
# PARSING-SPECIFIC VALIDATIONS
# ---------------------------------------------------------


def validate_relaxation_files(directory):
    """
    Validate files required for relaxation parsing.

    Parameters
    ----------
    directory : Path
        Path to the relaxation output directory.

    Returns
    -------
    bool
        True if all required relaxation files are present, False otherwise.
    """
    logger.info("Validating relaxation files...")
    return validate_required_files(DEFAULT_RELAXATION_FILES, directory)


def validate_optics_files(directory):
    """
    Validate files required for optics parsing.

    Parameters
    ----------
    directory : Path
        Path to the optics output directory.

    Returns
    -------
    bool
        True if all required optics files are present, False otherwise.
    """
    logger.info("Validating optics files...")
    return validate_required_files(DEFAULT_OPTICS_FILES, directory)


def validate_artatop_outputs(directory):
    """
    Validate outputs required for ARTATOP parsing.

    Parameters
    ----------
    directory : Path
        Path to the ARTATOP output directory.

    Returns
    -------
    bool
        True if all required ARTATOP outputs (files and directories) are present, False otherwise.
    """
    logger.info("Validating ARTATOP outputs...")
    files_valid = validate_required_files(DEFAULT_ARTATOP_OUTPUT_FILES, directory)
    dirs_valid = validate_required_dirs(DEFAULT_ARTATOP_OUTPUT_DIRS, directory)
    return files_valid and dirs_valid
