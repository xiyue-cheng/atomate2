"""Module for defining ARTATOP jobs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Maker, job

from atomate2 import SETTINGS
from atomate2.common.files import gzip_output_folder
from atomate2.artatop.sets.core import InputFileHandler
from atomate2.artatop.files import (
    ARTATOP_OUTPUT_FILES,
    VASP_OUTPUT_FILES,
    copy_artatop_files,
    validate_required_files,
)
from atomate2.artatop.run import run_artatop
from atomate2.artatop.schemas import ArtatopTaskDocument

logger = logging.getLogger(__name__)

_FILES_TO_ZIP = [*ARTATOP_OUTPUT_FILES, *VASP_OUTPUT_FILES]


@dataclass
class ARTATOPMaker(Maker):
    """
    ARTATOP job maker.

    The maker copies DFT output files necessary for the ARTATOP run. It generates
    ARTATOP input files using `InputFileHandler`, runs ARTATOP, compresses outputs,
    and parses the results.

    Parameters
    ----------
    name : str
        Name of jobs produced by this maker.
    task_document_kwargs : dict
        Keyword arguments passed to :obj:`.ArtatopTaskDocument.from_directory`.
    run_artatop_kwargs : dict
        Keyword arguments passed to :obj:`.run_artatop`.
    calc_type : str
        Calculation type ("lin", "nlin", "art").
    custom_components : str
        Optional components for ART calculations (e.g., tensor components).
    """

    name: str = "artatop"
    task_document_kwargs: dict = field(default_factory=dict)
    run_artatop_kwargs: dict = field(default_factory=dict)
    calc_type: str = "lin"
    custom_components: str | None = None

    @job(output_schema=ArtatopTaskDocument)
    def make(
        self, 
        wavefunction_dir: str | Path = None,
    ) -> ArtatopTaskDocument:
        """
        Run an ARTATOP calculation.

        Parameters
        ----------
        wavefunction_dir : str or Path
            A directory containing a WAVEFUNCTION and other outputs needed for Lobster
        -------
        ArtatopTaskDocument
            Parsed results from ARTATOP calculations.
        """


        
        run_dir = Path.cwd()
        # Copy required files # VASP for example
        copy_artatop_files(wavefunction_dir, dest_dir=run_dir)
        
        # Validate required VASP output files in wavefunction_dir
        #validate_required_files(VASP_OUTPUT_FILES, Path(wavefunction_dir))
        #logger.info(f"All required files found in 'wavefunction_dir': {wavefunction_dir}")

        # Create input files for ARTATOP
        input_handler = InputFileHandler(output_dir="./artatop_outputs")
        artatop_input = input_handler.get_input_set(
            calc_type=self.calc_type,
            calc_dir=Path.cwd(),
            component=self.custom_components,
        )

        # Run ARTATOP
        logger.info(f"Running ARTATOP for {self.calc_type} calculation")
        run_artatop(
            artatop_cmd=f"artatop < {artatop_input} > output_{self.calc_type}",
            **self.run_artatop_kwargs,
        )

        # Compress output files
        logger.info("Compressing ARTATOP output files")
        gzip_output_folder(
            directory=Path.cwd(),
            setting=SETTINGS.ARTATOP_ZIP_FILES,
            files_list=_FILES_TO_ZIP,
        )

        # Parse outputs
        logger.info("Parsing ARTATOP outputs")
        return ArtatopTaskDocument.from_directory(
            dir_name=str(Path.cwd()), **self.task_document_kwargs
        )
