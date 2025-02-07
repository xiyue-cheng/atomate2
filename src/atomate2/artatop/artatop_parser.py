"""Module provides parsing functions for ARTATOP outputs."""

import logging
from pathlib import Path

from atomate2.artatop.schemas import (
    AtomicContributions,
    LinearOpticalResponse,
    NonlinearOpticalResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# PARSING FUNCTIONS
# ---------------------------------------------------------


def parse_linear_response(out_lin_dir: Path) -> list[LinearOpticalResponse]:
    """
    Parse all linear optical response files in the `out_lin` directory.

    Parameters
    ----------
    out_lin_dir : Path
        Directory containing linear optical response files.

    Returns
    -------
    List[LinearOpticalResponse]
        List of parsed linear optical response data.
    """
    responses = []
    for file in out_lin_dir.glob("lin_*.dat"):
        logger.info(f"Parsing linear response file: {file}")
        with open(file) as f:
            lines = f.readlines()

        for line in lines:
            if not line.startswith("#") and line.strip():
                parts = line.split()
                response = LinearOpticalResponse(
                    energy=float(parts[0]),
                    real_part=float(parts[1]),
                    imaginary_part=float(parts[2]),
                    absorption_coefficient=float(parts[3]) if len(parts) > 3 else None,
                    refractive_index=float(parts[4]) if len(parts) > 4 else None,
                    extinction_coefficient=float(parts[5]) if len(parts) > 5 else None,
                )
                responses.append(response)
    return responses


def parse_nonlinear_response(out_nonlin_dir: Path) -> list[NonlinearOpticalResponse]:
    """
    Parse all nonlinear optical response files in the `out_nonlin` directory.

    Parameters
    ----------
    out_nonlin_dir : Path
        Directory containing nonlinear optical response files.

    Returns
    -------
    List[NonlinearOpticalResponse]
        List of parsed nonlinear optical response data.
    """
    responses = []
    for file in out_nonlin_dir.glob("nonlin_*.txt"):
        logger.info(f"Parsing nonlinear response file: {file}")
        with open(file) as f:
            lines = f.readlines()

        for line in lines:
            if not line.startswith("#") and line.strip():
                parts = line.split()
                response = NonlinearOpticalResponse(
                    energy=float(parts[0]),
                    total_real_part=float(parts[1]),
                    total_imaginary_part=float(parts[2]),
                    contributions={
                        "real": float(parts[3]) if len(parts) > 3 else None,
                        "imaginary": float(parts[4]) if len(parts) > 4 else None,
                    },
                )
                responses.append(response)
    return responses


def parse_atomic_contributions(out_nonlin_dir: Path) -> list[AtomicContributions]:
    """
    Parse atomic contributions from the `out_nonlin` directory.

    Parameters
    ----------
    out_nonlin_dir : Path
        Directory containing atomic contributions data.

    Returns
    -------
    List[AtomicContributions]
        List of parsed atomic contributions data.
    """
    contributions: list[AtomicContributions] = []  # Type annotation added
    atomic_file = out_nonlin_dir / "arp_nonlin.txt"
    if not atomic_file.exists():
        logger.warning(f"Atomic contributions file missing: {atomic_file}")
        return contributions

    logger.info(f"Parsing atomic contributions file: {atomic_file}")
    with open(atomic_file) as f:
        lines = f.readlines()

    for line in lines:
        if not line.startswith("#") and line.strip():
            parts = line.split()
            contribution = AtomicContributions(
                atom=parts[0],
                orbital_contributions={
                    "s": float(parts[1]),
                    "p": float(parts[2]),
                    "d": float(parts[3]),
                },
                total_contribution=float(parts[4]),
            )
            contributions.append(contribution)  # Add to the list
    return contributions


# ---------------------------------------------------------
# MAIN PARSING FUNCTION
# ---------------------------------------------------------


def parse_artatop_outputs(dir_name: str) -> dict:
    """
    Parse ARTATOP outputs from the specified directory and include metadata.

    Parameters
    ----------
    dir_name : str
        Directory containing ARTATOP output files.

    Returns
    -------
    dict
        Parsed data including linear response, nonlinear response,
        atomic contributions, and metadata.
    """
    from atomate2.artatop.parsers import (
        parse_atomic_contributions,
        parse_linear_response,
        parse_nonlinear_response,
    )

    dir_path = Path(dir_name)

    try:
        # Parse individual components
        linear_response = parse_linear_response(dir_path / "out_lin")
        nonlinear_response = parse_nonlinear_response(dir_path / "out_nonlin")
        atomic_contributions = (
            parse_atomic_contributions(dir_path / "out_nonlin")
            if (dir_path / "out_nonlin" / "arp_nonlin.txt").exists()
            else None
        )
    except Exception:
        logging.exception(f"Failed to parse ARTATOP outputs in {dir_name}")
        raise
    else:
        # Collect metadata
        metadata = {"dir_name": str(dir_path)}

        # Return parsed data with metadata
        return {
            **metadata,
            "linear_response": linear_response,
            "nonlinear_response": nonlinear_response,
            "atomic_contributions": atomic_contributions,
        }
