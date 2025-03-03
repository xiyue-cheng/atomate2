"""
Module defines job makers for ARTATOP workflows.

It includes:
- LINMaker: For linear optical response calculations.
- NLINMaker: For nonlinear optical response calculations.
- ARTMaker: For atomic response calculations.
"""
from __future__ import annotations
import logging
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from pathlib import Path
from copy import deepcopy

from jobflow import Flow, Response, job
from atomate2.utils.path import strip_hostname
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.common.files import copy_files
from atomate2.vasp.sets.core import VaspInputGenerator
from atomate2.artatop.jobs import ARTATOPMaker
from atomate2.artatop.sets.core import InputFileHandler

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class ARTATOPStaticMaker(BaseVaspMaker):
    """
    Maker for performing a static calculation required for ARTATOP.
    Uses default settings unless overridden.
    """
    name: str = "artatop_static"
    input_set_generator: VaspInputGenerator = field(default_factory=VaspInputGenerator)

@job
def run_artatop_static(structure: Structure) -> Response:
    """
    Run the static calculation required for ARTATOP without explicitly providing an INCAR.
    """
    static_maker = ARTATOPStaticMaker()
    static_job = static_maker.make(structure=structure)
    return static_job

class ARTATOPOpticsMaker(BaseVaspMaker):
    """
    Maker for performing an optics calculation required for ARTATOP.
    """
    name: str = "optics"
    input_set_generator: VaspInputGenerator = field(default_factory=VaspInputGenerator)

@job
def run_artatop_optics(
    structure: Structure,
    prev_dir: Path | str,
) -> Response:
    """
    Run the ARTATOP optics job to prepare input files for LIN, NLIN, ART.

    Parameters
    ----------
    structure : Structure
        The structure for the optics job.
    prev_dir : Path or str
        The previous VASP calculation directory.

    Returns
    -------
    Response
        The directory containing the optics calculation results.
    """
    prev_dir = Path(prev_dir)
    optics_maker = ARTATOPOpticsMaker()
    
    optics_maker = ArtatopOpticsMaker()
    optics_job = optics_maker.make(structure=structure, prev_dir=prev_dir)
    optics_job.append_name("_optics")

    return Response(output={"optics_dir": optics_job.output.dir_name})


@job
def get_lin_jobs(
    artatop_maker: ARTATOPMaker | None,
    optics_dir: Path | str,
) -> Response:
    """
    Create LIN jobs for ARTATOP.

    Parameters
    ----------
    artatop_maker : ARTATOPMaker or None
        Maker for ARTATOP jobs.
    optics_dir : Path or str
        Directory with optics results.

    Returns
    -------
    Response
        A response containing the LIN job.
    """
    optics_dir = Path(optics_dir)
    if not optics_dir.exists():
        raise FileNotFoundError(f"Optics directory '{optics_dir}' does not exist!")

    jobs = []
    outputs = {
        "optics_dir": str(optics_dir),
        "lin_dirs": [],
        "lin_task_documents": [],
    }

    input_handler = InputFileHandler(output_dir=str(optics_dir))
    input_handler.get_input_set("lin", optics_dir)

    artatop_maker = artatop_maker or ARTATOPMaker(calc_type="lin")

    lin_job = artatop_maker.make(wavefunction_dir=optics_dir)
    lin_job.append_name("_lin_calculation")

    outputs["lin_dirs"].append(lin_job.output.dir_name)
    outputs["lin_task_documents"].append(lin_job.output)
    jobs.append(lin_job)

    return Response(replace=Flow(jobs, output=outputs))


@job
def get_nlin_jobs(
    artatop_maker: ARTATOPMaker | None,
    optics_dir: Path | str,
) -> Response:
    """
    Create NLIN jobs for ARTATOP.

    Parameters
    ----------
    artatop_maker : ARTATOPMaker or None
        Maker for ARTATOP jobs.
    optics_dir : Path or str
        Directory with optics results.

    Returns
    -------
    Response
        A response containing the NLIN job.
    """
    optics_dir = Path(optics_dir)
    if not optics_dir.exists():
        raise FileNotFoundError(f"Optics directory '{optics_dir}' does not exist!")

    jobs = []
    outputs = {
        "optics_dir": str(optics_dir),
        "nlin_dirs": [],
        "nlin_task_documents": [],
    }

    input_handler = InputFileHandler(output_dir=str(optics_dir))
    input_handler.get_input_set("nlin", optics_dir)

    artatop_maker = artatop_maker or ARTATOPMaker(calc_type="nlin")

    nlin_job = artatop_maker.make(wavefunction_dir=optics_dir)
    nlin_job.append_name("_nlin_calculation")

    outputs["nlin_dirs"].append(nlin_job.output.dir_name)
    outputs["nlin_task_documents"].append(nlin_job.output)
    jobs.append(nlin_job)

    return Response(replace=Flow(jobs, output=outputs))


@job
def get_art_jobs(
    artatop_maker: ARTATOPMaker | None,
    optics_dir: Path | str,
) -> Response:
    """
    Create ART jobs for ARTATOP.

    Parameters
    ----------
    artatop_maker : ARTATOPMaker or None
        Maker for ARTATOP jobs.
    optics_dir : Path or str
        Directory with optics results.

    Returns
    -------
    Response
        A response containing the ART job.
    """
    optics_dir = Path(optics_dir)
    if not optics_dir.exists():
        raise FileNotFoundError(f"Optics directory '{optics_dir}' does not exist!")

    jobs = []
    outputs = {
        "optics_dir": str(optics_dir),
        "art_dirs": [],
        "art_task_documents": [],
    }

    input_handler = InputFileHandler(output_dir=str(optics_dir))
    result_re_file = optics_dir / "result.re"

    if not result_re_file.exists():
        raise FileNotFoundError(f"Expected ARTATOP output `{result_re_file}` not found!")

    highest_component = input_handler.determine_highest_component(result_re_file)
    input_handler.get_input_set("art", optics_dir, component=highest_component)

    artatop_maker = artatop_maker or ARTATOPMaker(calc_type="art")

    art_job = artatop_maker.make(wavefunction_dir=optics_dir)
    art_job.append_name(f"_art_calculation_{highest_component}")

    outputs["art_dirs"].append(art_job.output.dir_name)
    outputs["art_task_documents"].append(art_job.output)
    jobs.append(art_job)

    return Response(replace=Flow(jobs, output=outputs))                                                                  
