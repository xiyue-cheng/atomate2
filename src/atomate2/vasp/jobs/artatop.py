"""
Module defines job makers for ARTATOP workflows.

It includes:
- LINMaker: For linear optical response calculations.
- NLINMaker: For nonlinear optical response calculations.
- ARTMaker: For atomic response calculations.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jobflow import Maker, Response, job
from atomate2.artatop.schemas import ArtatopTaskDocument, ArtatopInputModel, ArtatopOutputModel

from atomate2.artatop.run import run_artatop
from atomate2.artatop.sets.core import InputFileHandler
from atomate2.utils.path import strip_hostname
from atomate2.artatop.jobs import ARTATOPMaker


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)

import os
import shutil
from pathlib import Path


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
        
        # Set input and output files
        input_lin = job_dir / "input_lin"
        lin_output = job_dir / "re_lin"

        # Ensure lin_output is valid and create necessary files
        self.input_handler = InputFileHandler(output_dir=str(out_lin))
        self.input_handler.get_input_set("lin", job_dir)  # job_dir is already Path

        command = f"artatop < {input_lin} > {lin_output}"
        run_artatop(job_type="direct", artatop_cmd=command)
        
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

        # Input and output files in job_dir
        input_nlin = job_dir / "input_nlin"
        nlin_output = job_dir / "re_nlin"  # Outputs stored in `out_nonlin`

        # Generate input files for NLIN
        self.input_handler = InputFileHandler(output_dir=str(out_nonlin))
        self.input_handler.get_input_set("nlin", job_dir)

        # Run ARTATOP NLIN calculation
        command = f"artatop < {input_nlin} > {nlin_output}"
        run_artatop(job_type="direct", artatop_cmd=command)

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
        output_dir = job_dir / "out_nlin"
        output_dir.mkdir(parents=True, exist_ok=True)

        input_art = prev_dir / "input_art"
        self.input_handler = InputFileHandler(
            output_dir=str(prev_dir)
        )  # Convert to str
        self.input_handler.get_input_set("art", prev_dir)

        art_output = prev_dir / "re_art"
        command = f"artatop < {input_art} > {art_output}"
        run_artatop(job_type="direct", artatop_cmd=command)

        return {"art_output": str(art_output)}
