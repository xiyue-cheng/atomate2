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

from jobflow import Flow, Response, job
from atomate2.utils.path import strip_hostname
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.core import NonSCFSetGenerator
from atomate2.artatop.jobs import ARTATOPMaker
from atomate2.artatop.sets.core import InputFileHandler

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class ArtatopOpticsMaker(BaseVaspMaker):
    """
    Maker that performs a VASP NonSCF computation with settings required for ARTATOP.

    This is equivalent to `LobsterStaticMaker` but tailored for ARTATOP.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : NonSCFSetGenerator
        A generator used to make the input set.
    """

    name: str = "artatop_optics_run"
    input_set_generator: NonSCFSetGenerator = field(
        default_factory=lambda: NonSCFSetGenerator(
            user_kpoints_settings={"reciprocal_density": 400},
            user_incar_settings={
                "LOPTICS": True,  # Optical calculations
                "LWAVE": True,  # Save wavefunction
                "ISYM": 0,  # No symmetry for accuracy
                "NBANDS": 200,  # Sufficient bands
                "ALGO": "Exact",
                "EDIFF": 1e-6,
                "LREAL": False,
                "NCORE": 4,
            },
        )
    )


@job
def get_artatop_optics(
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
    optics_dir = Path.cwd()
    optics_maker = ArtatopOpticsMaker()
    optics_job = optics_maker.make(structure=structure, prev_dir=prev_dir)

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

    art_job = artatop_maker.make(input_dir=optics_dir)
    art_job.append_name(f"_art_calculation_{highest_component}")

    outputs["art_dirs"].append(art_job.output.dir_name)
    outputs["art_task_documents"].append(art_job.output)
    jobs.append(art_job)

    return Response(replace=Flow(jobs, output=outputs))                                                                  
