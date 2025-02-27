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


@job
def get_lin_jobs(
    artatop_maker: ARTATOPMaker | None,
    optics_dir: Path | str,
) -> Response:
    """
    Create LIN jobs for ARTATOP, ensuring proper input files are generated.

    Parameters
    ----------
    artatop_maker : ARTATOPMaker or None
        Maker for the ARTATOP jobs.
    optics_dir : Path or str
        Directory containing the required VASP outputs (OPTICS output).

    Returns
    -------
    Response
        A response containing the LIN job.
    """
    optics_dir = Path(optics_dir)
    if not optics_dir.exists():
        raise FileNotFoundError(f"Optics directory '{optics_dir}' does not exist!")

    jobs = []
    outputs: dict[str, Any] = {
        "optics_dir": str(optics_dir),
        "lin_dirs": [],
        "lin_task_documents": [],
    }

    # **Generate ARTATOP input file for LIN calculation**
    input_handler = InputFileHandler(output_dir=str(optics_dir))
    input_handler.get_input_set("lin", optics_dir)

    artatop_maker = artatop_maker or ARTATOPMaker(calc_type="lin")

    lin_job = artatop_maker.make(input_dir=optics_dir)
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
    Create NLIN jobs for ARTATOP, ensuring proper input files are generated.
    """
    optics_dir = Path(optics_dir)
    if not optics_dir.exists():
        raise FileNotFoundError(f"Optics directory '{optics_dir}' does not exist!")

    jobs = []
    outputs: dict[str, Any] = {
        "optics_dir": str(optics_dir),
        "nlin_dirs": [],
        "nlin_task_documents": [],
    }

    # **Generate ARTATOP input file for NLIN calculation**
    input_handler = InputFileHandler(output_dir=str(optics_dir))
    input_handler.get_input_set("nlin", optics_dir)

    artatop_maker = artatop_maker or ARTATOPMaker(calc_type="nlin")

    nlin_job = artatop_maker.make(input_dir=optics_dir)
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
    Create ART jobs for ARTATOP, ensuring proper input files are generated.
    """
    optics_dir = Path(optics_dir)
    if not optics_dir.exists():
        raise FileNotFoundError(f"Optics directory '{optics_dir}' does not exist!")

    jobs = []
    outputs: dict[str, Any] = {
        "optics_dir": str(optics_dir),
        "art_dirs": [],
        "art_task_documents": [],
    }

    # **Determine the highest tensor component from result.re**
    input_handler = InputFileHandler(output_dir=str(optics_dir))
    result_re_file = optics_dir / "result.re"

    if not result_re_file.exists():
        raise FileNotFoundError(f"Expected ARTATOP output file `{result_re_file}` not found!")

    highest_component = input_handler.determine_highest_component(result_re_file)

    # **Generate ART input file with highest component**
    input_handler.get_input_set("art", optics_dir, component=highest_component)

    artatop_maker = artatop_maker or ARTATOPMaker(calc_type="art")

    art_job = artatop_maker.make(input_dir=optics_dir)
    art_job.append_name(f"_art_calculation_{highest_component}")

    outputs["art_dirs"].append(art_job.output.dir_name)
    outputs["art_task_documents"].append(art_job.output)
    jobs.append(art_job)

    return Response(replace=Flow(jobs, output=outputs))
