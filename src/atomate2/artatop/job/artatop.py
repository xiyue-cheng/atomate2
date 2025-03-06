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

from jobflow import Maker, Response, job, Flow
from atomate2.artatop.sets.core import InputFileHandler
from atomate2.common.files import copy_files

from atomate2.utils.path import strip_hostname
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.core import NonSCFSetGenerator
from atomate2.artatop.jobs import ARTATOPMaker
from atomate2.artatop.schemas import ArtatopTaskDocument, ArtatopInputModel, ArtatopOutputModel

if TYPE_CHECKING:
    from pathlib import Path
    from pymatgen.core import Structure
    from atomate2.vasp.sets.base import VaspInputGenerator
logger = logging.getLogger(__name__)

from pathlib import Path  # Ensure Path is imported

@job
def get_artatop_jobs(
    optics_job_output,  # This should be the output of the optics_flow (e.g., TaskDoc)
    artatop_maker: ARTATOPMaker | None = None,
) -> Response:
    """
    Create a list of ARTATOP jobs (LIN, NLIN, ART) dynamically.

    Parameters
    ----------
    optics_job_output : TaskDoc
        The output reference from OpticsMaker, which is a TaskDoc object.
    artatop_maker : ARTATOPMaker or None
        Maker for the ARTATOP jobs.

    Returns
    -------
    Response
        A response containing the ARTATOP jobs.
    """
    jobs = []
    outputs = {
        "optics_dir": optics_job_output.dir_name,  # Extract the directory path from TaskDoc
        "artatop_dirs": [],
        "artatop_task_documents": [],
    }

    artatop_maker = artatop_maker or ARTATOPMaker()

    # Loop over ARTATOP calculation types (LIN, NLIN, ART)
    for idx, calc_type in enumerate(["lin", "nlin", "art"]):
        # Create a unique directory for each ARTATOP calculation
        calc_dir = Path(f"artatop_{calc_type}_{idx}")  # Ensure calc_dir is a Path object
        calc_dir.mkdir(parents=True, exist_ok=True)  # Create the directory

        # Copy required files from the optics directory to the new ARTATOP directory
        copy_artatop_files(src_dir=optics_job_output.dir_name, dest_dir=calc_dir)

        # Create input files for ARTATOP in the new directory
        input_handler = InputFileHandler(output_dir=calc_dir)
        input_handler.get_input_set(calc_type, calc_dir)

        # Run ARTATOP in the new directory
        artatop_job = artatop_maker.make(wavefunction_dir=calc_dir, calc_type=calc_type)
        artatop_job.append_name(f"_{calc_type}_calculation_{idx}")

        # Store job details
        outputs["artatop_dirs"].append(str(calc_dir))  # Store as string if needed
        outputs["artatop_task_documents"].append(artatop_job.output)
        jobs.append(artatop_job)
        print(f"calc_dir: {calc_dir}, type: {type(calc_dir)}")

    # Return all jobs as a Flow
    return Response(replace=Flow(jobs, output=outputs))
