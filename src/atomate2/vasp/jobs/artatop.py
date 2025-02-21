"""
Module defining ARTATOP job makers.

It includes:
- LINMaker: For linear optical response calculations.
- NLINMaker: For nonlinear optical response calculations.
- ARTMaker: For atomic response calculations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jobflow import Response, job
from atomate2.artatop.jobs import ARTATOPMaker

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LINMaker:
    """Maker for ARTATOP Linear (LIN) calculations."""

    name: str = "ARTATOP LIN Maker"

@job
def get_lin_job(prev_dir: Path | str) -> Response:
    """
    Define the ARTATOP LIN job.

    Parameters
    ----------
    prev_dir : Path or str
        Directory of previous calculations.

    Returns
    -------
    Response
        A response containing the ARTATOP LIN job.
    """
    artatop_maker = ARTATOPMaker(job_type="lin")  # Create LIN job
    lin_job = artatop_maker.make(prev_dir=prev_dir)  # Define execution
    return Response(replace=lin_job)  # Return the LIN job


@dataclass
class NLINMaker:
    """Maker for ARTATOP Non-Linear (NLIN) calculations."""

    name: str = "ARTATOP NLIN Maker"

@job
def get_nlin_job(prev_dir: Path | str) -> Response:
    """
    Define the ARTATOP NLIN job.

    Parameters
    ----------
    prev_dir : Path or str
        Directory of previous calculations.

    Returns
    -------
    Response
        A response containing the ARTATOP NLIN job.
    """
    artatop_maker = ARTATOPMaker(job_type="nlin")  # Create NLIN job
    nlin_job = artatop_maker.make(prev_dir=prev_dir)  # Define execution
    return Response(replace=nlin_job)  # Return the NLIN job


@dataclass
class ARTMaker:
    """Maker for ARTATOP ART calculations."""

    name: str = "ARTATOP ART Maker"

@job
def get_art_job(prev_dir: Path | str) -> Response:
    """
    Define the ARTATOP ART job.

    Parameters
    ----------
    prev_dir : Path or str
        Directory of previous calculations.

    Returns
    -------
    Response
        A response containing the ARTATOP ART job.
    """
    artatop_maker = ARTATOPMaker(job_type="art")  # Create ART job
    art_job = artatop_maker.make(prev_dir=prev_dir)  # Define execution
    return Response(replace=art_job)  # Return the ART job
