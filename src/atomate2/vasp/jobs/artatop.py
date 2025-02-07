"""Module Defining Job Makers for ARTATOP workflow."""

from __future__ import annotations

import logging
from pathlib import Path

from jobflow import Maker, Response, job

from atomate2.artatop.run import run_artatop
from atomate2.artatop.sets.core import InputFileHandler

logger = logging.getLogger(__name__)


class LINMaker(Maker):
    """Maker for ARTATOP Linear (LIN) calculations."""

    name: str = "ARTATOP LIN Maker"

    def __init__(self) -> None:
        """Initialize the LINMaker class."""
        self.input_handler: InputFileHandler  # Properly annotate the type

    @job
    def make(self, prev_dir: Path | str, job_dir: Path | str) -> Response:
        """
        Run LIN (linear) calculations.

        Parameters
        ----------
        prev_dir : str
            Directory containing optics output.
        job_dir : str
            Base directory for storing LIN outputs.

        Returns
        -------
        Response
        A jobflow Response containing the output directory for the LIN calculation.
        """
        prev_dir = Path(prev_dir)
        job_dir = Path(job_dir)

        out_lin = job_dir / "out_lin"
        out_lin.mkdir(parents=True, exist_ok=True)

        # Required files to copy from prev_dir
        required_files = ["INCAR.gz", "CONTCAR.gz", "OPTIC", "PROCAR.gz", "WAVEDER.gz"]
        for file_name in required_files:
            source_file = prev_dir / file_name
            if not source_file.exists():
                raise FileNotFoundError(
                    f"Required file {file_name} not found in {prev_dir}."
                )

        self.input_handler = InputFileHandler(output_dir=str(job_dir))
        lin_input = self.input_handler.get_input_set("lin", job_dir)
        lin_output = job_dir / "re_lin"

        command = f"artatop < {lin_input} > {lin_output}"
        run_artatop(job_type="direct", artatop_cmd=command)

        return {"lin_output": str(lin_output)}


class NLINMaker(Maker):
    """Maker for ARTATOP Non-Linear (NLIN) calculations."""

    name: str = "ARTATOP NLIN Maker"

    def __init__(self) -> None:
        """Initialize the NLINMaker class."""
        self.input_handler: InputFileHandler  # Properly annotate the type

    @job
    def make(self, prev_dir: Path | str, job_dir: Path | str) -> Response:
        """
        Run NLIN (non-linear) calculations.

        Parameters
        ----------
        prev_dir : str
            Directory containing optics output.
        job_dir : str
            Base directory for storing NLIN outputs.

        Returns
        -------
        Response
        A jobflow Response containing the output directory for the NLIN calculation.
        """
        prev_dir = Path(prev_dir)
        job_dir = Path(job_dir)

        out_nonlin = job_dir / "out_nonlin"
        out_nonlin.mkdir(parents=True, exist_ok=True)

        # Required files to copy from prev_dir
        required_files = ["INCAR.gz", "CONTCAR.gz", "OPTIC", "PROCAR.gz", "WAVEDER.gz"]
        for file_name in required_files:
            source_file = prev_dir / file_name
            if not source_file.exists():
                raise FileNotFoundError(
                    f"Required file {file_name} not found in {prev_dir}."
                )

        self.input_handler = InputFileHandler(output_dir=str(job_dir))
        nlin_input = self.input_handler.get_input_set("nlin", job_dir)
        nlin_output = job_dir / "re_nlin"

        command = f"artatop < {nlin_input} > {nlin_output}"
        run_artatop(job_type="direct", artatop_cmd=command)

        return {"nlin_output": str(nlin_output)}


class ARTMaker(Maker):
    """Maker for ARTATOP ART calculations."""

    name: str = "ARTATOP ART Maker"

    def __init__(self) -> None:
        """Initialize the ARTMaker class."""
        self.input_handler: InputFileHandler  # Properly annotate the type

    @job
    def make(self, prev_dir: Path | str, job_dir: Path | str) -> Response:
        """
        Run ART calculations.

        Parameters
        ----------
        prev_dir : str
            Directory containing optics output.
        job_dir : str
            Base directory for storing ART outputs.

        Returns
        -------
        Response
        A jobflow Response containing the output directory for the ART calculation.
        """
        prev_dir = Path(prev_dir)
        job_dir = Path(job_dir)

        out_nonlin = job_dir / "out_nonlin"
        out_nonlin.mkdir(parents=True, exist_ok=True)

        self.input_handler = InputFileHandler(output_dir=str(job_dir))
        component = self.input_handler.determine_highest_component(job_dir)
        art_input = self.input_handler.get_input_set("art", job_dir, component)
        art_output = job_dir / "re_art"

        command = f"artatop < {art_input} > {art_output}"
        run_artatop(job_type="direct", artatop_cmd=command)

        return {"art_output": str(art_output)}
