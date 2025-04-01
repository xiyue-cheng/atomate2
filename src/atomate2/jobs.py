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

from atomate2.artatop.utils import run_artatop_shell_scripts
from atomate2.artatop.run import run_artatop
from atomate2.artatop.schemas import ArtatopTaskDocument


logger = logging.getLogger(__name__)

_FILES_TO_ZIP = [*ARTATOP_OUTPUT_FILES, *VASP_OUTPUT_FILES]


@dataclass
class ARTATOPMaker(Maker):
    """
    ARTATOP job makeomeger.

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
        wavefunction_dir = Path(wavefunction_dir)
        # Copy required files # VASP for example
        input_handler = InputFileHandler()
        copy_artatop_files(wavefunction_dir, dest_dir=run_dir)
        
        # Validate required VASP output files in wavefunction_dir
        #validate_required_files(VASP_OUTPUT_FILES, Path(wavefunction_dir))
        #logger.info(f"All required files found in 'wavefunction_dir': {wavefunction_dir}")
        
        # Create input files for ARTATOP
        
        
        def run_artatop_step(calc_type: str, input_file: Path):
            output_file = f"re_{calc_type}"
            stderr_file = f"std_err_{calc_type}.txt"
            artatop_cmd = f"{SETTINGS.ARTATOP_CMD} < {input_file} > {output_file}"
            print(f"Running ARTATOP command for {calc_type}:", artatop_cmd)
            
            from custodian.artatop.handlers import ArtatopFilesValidator

            self.run_artatop_kwargs["validators"] = [ArtatopFilesValidator(required_files=[output_file])]
           
            


            self.run_artatop_kwargs["artatop_job_kwargs"] = {
                "output_file": output_file,
                "stderr_file": stderr_file,
                "artatop_cmd": artatop_cmd,
                "gzipped": False
            }
            
            

            run_artatop(**self.run_artatop_kwargs)

            if not Path(output_file).exists():
                raise RuntimeError(f"ARTATOP failed: {output_file} not found.")
              
        # Step 2: Linear
        lin_input = input_handler.get_input_set("lin", calc_dir=run_dir)
        run_artatop_step("lin", lin_input)

        # Step 3: Nonlinear
        nlin_input = input_handler.get_input_set("nlin", calc_dir=run_dir)
        run_artatop_step("nlin", nlin_input)
        
        #run_artatop_shell_scripts(run_dir)

        # Step 4: Parse outputs and write result.re
        from atomate2.artatop.artatop_parser import parse_lin_nlin_outputs, write_result_re, parse_artatop_outputs
        output_model = parse_lin_nlin_outputs(run_dir)
        write_result_re(output_model, filename=run_dir / "result.re")

        # Step 4: Determine ART component (result.re or nonlin files)
        component = input_handler.determine_highest_component(run_dir / "result.re")

        # Step 5: ART
        art_input = input_handler.get_input_set("art", calc_dir=run_dir, component=component)
        run_artatop_step("art", art_input)
        
        self.task_document_kwargs["input_file"] = str(art_input)
        
        output_model = parse_artatop_outputs(run_dir)
        
                # Step 3: List current directory and files before parsing
        print("Files in run_dir before parsing:")
        for f in run_dir.iterdir():
            print("  ", f.name)
            
        # Parse outputs
        logger.info("Parsing ARTATOP outputs")

        # Step 4: Parse outputs
        print("Parsing ARTATOP outputs now...")
        
        print("Running ArtatopTaskDocument.from_directory...")
        try:
            doc = ArtatopTaskDocument.from_directory(dir_name=str(Path.cwd()), **self.task_document_kwargs
            )
            print("Parsing complete.")
        except Exception as e:
            print("FAILED during from_directory:", e)
            raise e

        # Save readable output
        from monty.serialization import dumpfn
        
        print("Saving artatop_result.json now...")
        output_path = Path.cwd() / "artatop_result.json"
        dumpfn(doc.dict(), output_path, indent=4)
        print("Saved successfully.")

        
        
         # Compress output files
        logger.info("Compressing ARTATOP output files")
        
        return doc
