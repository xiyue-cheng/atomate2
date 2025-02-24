"""
Module defines job makers for ARTATOP workflows.

It includes:
- LINMaker: For linear optical response calculations.
- NLINMaker: For nonlinear optical response calculations.
- ARTMaker: For atomic response calculations.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path


import logging

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jobflow import Maker, Response, job
from atomate2.artatop.sets.core import InputFileHandler

from atomate2.artatop.jobs import ARTATOPMaker
from atomate2.utils.path import strip_hostname
from atomate2.artatop.schemas import ArtatopTaskDocument, ArtatopInputModel, ArtatopOutputModel

if TYPE_CHECKING:
    from pathlib import Path
    from pymatgen.core import Structure
    from atomate2.vasp.sets.base import VaspInputGenerator
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
    artatop_maker = ARTATOPMaker()  # Create LIN job
    lin_job = artatop_maker.make(prev_dir=prev_dir, job_type="nlin")  # Define execution
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
    artatop_maker = ARTATOPMaker()  # Create NLIN job
    nlin_job = artatop_maker.make(prev_dir=prev_dir, job_type="nlin")  # Define execution
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
    artatop_maker = ARTATOPMaker()  # Create ART job
    art_job = artatop_maker.make(prev_dir=prev_dir, job_type="art")  # Define execution
    return Response(replace=art_job)  # Return the ART job
