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




class LINMaker(Maker):
    """Maker for ARTATOP Linear (LIN) calculations."""

    name: str = "ARTATOP LIN Maker"

    @job
    def make(self, prev_dir: str) -> Response:
        # Ensure job_dir is a Path object and exists
        job_dir = Path.cwd()  # Get current working directory
 
        # Convert prev_dir to Path and ensure it's valid
        prev_dir = Path(prev_dir)
        
        # Prepare output directory for lin calculations
        out_lin = job_dir / "out_lin"
        out_lin.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
        print(f"Output directory for lin calculations: {out_lin}")
        
        input_lin = job_dir / "input_lin"
        lin_output = job_dir / "re_lin"
        
        # Return the response with lin_output
        return Response(output={"lin_output": str(lin_output)})

class NLINMaker(Maker):
    """Maker for ARTATOP Non-Linear (NLIN) calculations."""

    name: str = "ARTATOP NLIN Maker"

    @job
    def make(self, prev_dir: str) -> Response:
        # Main job directory
        job_dir = Path.cwd()

        # Ensure prev_dir exists
        prev_dir = Path(prev_dir)

        # Define the ARTATOP-specific output directory
        out_nonlin = job_dir / "out_nonlin"
        out_nonlin.mkdir(parents=True, exist_ok=True)
        
        input_nlin = job_dir / "input_nlin"
        nlin_output = job_dir / "re_nlin"

        
        # Return Response with output path
        return Response(output={"nlin_output": str(nlin_output)})


class ARTMaker(Maker):
    """Maker for ARTATOP ART calculations."""

    name: str = "ARTATOP ART Maker"

    @job
    def make(self, prev_dir: str) -> dict:
        prev_dir = Path(prev_dir)  # Ensure prev_dir is a Path
        prev_dir.mkdir(parents=True, exist_ok=True)

        job_dir = Path.cwd()  # Ensure job_dir is a Path
        
        art_output = prev_dir / "re_art"

        return {"art_output": str(art_output)}
