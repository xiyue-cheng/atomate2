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

from atomate2.utils.path import strip_hostname
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.core import OpticsSetGenerator
from atomate2.artatop.jobs import ARTATOPMaker
from atomate2.artatop.schemas import ArtatopTaskDocument, ArtatopInputModel, ArtatopOutputModel

if TYPE_CHECKING:
    from pathlib import Path
    from pymatgen.core import Structure
    from atomate2.vasp.sets.base import VaspInputGenerator
logger = logging.getLogger(__name__)

"""Module defining ARTATOP jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jobflow import Flow, Response, job


from atomate2.utils.path import strip_hostname
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.core import OpticsSetGenerator
from atomate2.artatop.jobs import ARTATOPMaker  # Assuming ARTATOPMaker exists

if TYPE_CHECKING:
    from pathlib import Path
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class OpticsStaticMaker(BaseVaspMaker):
    """
    Maker that performs a VASP optics computation required for ARTATOP.

    This runs:
    1. A static calculation
    2. A non-self-consistent optics calculation (LOPTICS=True)

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    """

    name: str = "optics_run"
    input_set_generator: OpticsSetGenerator = field(
        default_factory=lambda: OpticsSetGenerator(optics=True)
    )


@job
def get_artatop_jobs(
    artatop_maker: ARTATOPMaker,
    optics_dir: Path | str,
    optics_uuid: str,
) -> Response:
    """
    Create ARTATOP jobs.

    Parameters
    ----------
    artatop_maker : .ARTATOPMaker
        Maker for ARTATOP jobs.
    optics_dir : Path or str
        Path to the optics VASP calculation containing WAVEDER and OPTICS files.
    optics_uuid : str
        UUID of the optics calculation.

    Returns
    -------
    Response
        Flow of ARTATOP jobs.
    """
    jobs = []
    outputs: dict[str, Any] = {
        "optics_dir": optics_dir,
        "optics_uuid": optics_uuid,
        "artatop_uuids": [],
        "artatop_dirs": [],
        "artatop_task_documents": [],
    }

    artatop_job = artatop_maker.make(wavefunction_dir=optics_dir)
    artatop_job.append_name("_artatop")
    outputs["artatop_uuids"].append(artatop_job.output.uuid)
    outputs["artatop_dirs"].append(artatop_job.output.dir_name)
    outputs["artatop_task_documents"].append(artatop_job.output)
    jobs.append(artatop_job)

    flow = Flow(jobs, output=outputs)
    return Response(replace=flow)


@job
def delete_artatop_waveder(
    dirs: list[Path | str],
    optics_dir: Path | str = None,
) -> None:
    """
    Delete WAVEDER files after ARTATOP run.

    Parameters
    ----------
    dirs : list of path or str
        Path to directories of ARTATOP jobs.
    optics_dir : Path or str
        Path to directory of optics VASP run.
    """
    if optics_dir:
        dirs.append(optics_dir)

    for dir_name in dirs:
        delete_files(
            strip_hostname(dir_name),
            include_files=["WAVEDER", "WAVEDER.gz", "OPTICS"],
            allow_missing=True,
        )
