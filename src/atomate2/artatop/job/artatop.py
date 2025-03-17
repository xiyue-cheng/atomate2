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
from atomate2.artatop.files import copy_artatop_files
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
    outputs = {
        "optics_dir": optics_job_output.dir_name,  # Extract the directory path from TaskDoc
        "artatop_dirs": [],
        "artatop_task_documents": [],
    }
    
    #print(os.listdir(optics_job_output.dir_name))

    artatop_maker = artatop_maker or ARTATOPMaker()


    # Run ARTATOP in the new directory
    artatop_job = artatop_maker.make(wavefunction_dir=optics_job_output.dir_name)

    # Store job details
    outputs["artatop_dirs"].append(str(Path(".")))  # Store as string if needed
    outputs["artatop_task_documents"].append(artatop_job.output)
    #print(f"calc_dir: {calc_dir}, type: {type(calc_dir)}")

    # Return all jobs as a Flow
    return Response(replace=Flow(artatop_job, output=outputs))
