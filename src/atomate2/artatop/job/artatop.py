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
from atomate2.common.files import copy_files

from atomate2.artatop.jobs import ARTATOPMaker
from atomate2.utils.path import strip_hostname
from atomate2.artatop.schemas import ArtatopTaskDocument, ArtatopInputModel, ArtatopOutputModel

if TYPE_CHECKING:
    from pathlib import Path
    from pymatgen.core import Structure
    from atomate2.vasp.sets.base import VaspInputGenerator
logger = logging.getLogger(__name__)

@job
def get_artatop_jobs(
    artatop_maker: ARTATOPMaker | None,
    optics_job_output,  # This will dynamically resolve to the actual optics directory
) -> Response:
    """
    Create a list of ARTATOP jobs (LIN, NLIN, ART) dynamically.

    Parameters
    ----------
    artatop_maker : ARTATOPMaker or None
        Maker for the ARTATOP jobs.
    optics_job_output : JobFlow Output Reference
        The output reference from OpticsMaker, which resolves to the actual optics directory.

    Returns
    -------
    Response
        A response containing the ARTATOP jobs.
    """
    optics_job_output = str(optics_job_output)
    
    jobs = []
    outputs = {
        "optics_dir": optics_job_output,  # Store resolved optics_dir
        "artatop_dirs": [],
        "artatop_task_documents": [],
    }

    artatop_maker = artatop_maker or ARTATOPMaker()

    # Loop over ARTATOP calculation types (LIN, NLIN, ART)
    for idx, calc_type in enumerate(["lin", "nlin", "art"]):
        input_handler = InputFileHandler(output_dir=optics_job_output)
        input_handler.get_input_set(calc_type, optics_job_output)

        # Pass `optics_job_output` as the dynamically resolved wavefunction_dir
        artatop_job = artatop_maker.make(wavefunction_dir=optics_flow.output, calc_type=calc_type)
        artatop_job.append_name(f"_{calc_type}_calculation_{idx}")

        # Store job details
        outputs["artatop_dirs"].append(artatop_job.output.dir_name)
        outputs["artatop_task_documents"].append(artatop_job.output)
        jobs.append(artatop_job)

    # Return all jobs as a Flow
    return Response(replace=Flow(jobs, output=outputs))
