"""
Flows for ARTATOP computations.

This module provides a unified workflow for ARTATOP calculations combined
with VASP workflows, including relaxation, optics, and ARTATOP jobs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from pathlib import Path

from typing import Optional

from jobflow import Flow, Maker
from pymatgen.core import Structure
from atomate2.common.files import copy_files

from pymatgen.io.vasp import Poscar, Vasprun, Kpoints
from atomate2.vasp.flows.core import DoubleRelaxMaker, OpticsMaker
from atomate2.vasp.sets.core import RelaxSetGenerator, HSEStaticSetGenerator
from atomate2.vasp.jobs.core import RelaxMaker, HSEStaticMaker 
from atomate2.artatop.job.artatop import get_artatop_jobs
from atomate2.vasp.sets.core import NonSCFSetGenerator
from atomate2.artatop.job.artatop import ARTATOPMaker

try:
    import ijson
except ImportError:
    ijson = None

from atomate2.vasp.jobs.base import BaseVaspMaker

class ArtatopWorkflowMaker(Maker):
    """Workflow to run ARTATOP after VASP relax + optics + HSE."""

    name: str = "artatop_workflow"
    relax_maker: Maker = DoubleRelaxMaker()
    optics_maker: Maker = OpticsMaker()
    hse06_maker: Maker = HSEStaticMaker()
    

    def make(self, structure: Structure, prev_dir: str | Path, additional_metadata: Optional[dict] = None) -> Flow:
    
        relax_generator = RelaxSetGenerator(user_incar_settings={"NPAR": 4})
        relax1 = RelaxMaker(input_set_generator=relax_generator)
        relax2 = RelaxMaker(input_set_generator=relax_generator)
        relax_maker = DoubleRelaxMaker(relax_maker1=relax1, relax_maker2=relax2)

        self.optics_maker.static_maker.input_set_generator.user_incar_settings = {"LORBIT": 10, "NPAR": 4}
        self.optics_maker.band_structure_maker.input_set_generator = NonSCFSetGenerator(
            optics=True,
            nbands_factor=4.0,
            user_incar_settings={"LORBIT": 10, "CSHIFT": 0.1, "NPAR": 4, "ALGO": "Normal", "LMIXTAU": None},
        )
        
        hse_generator = HSEStaticSetGenerator(user_incar_settings={"NPAR": 4})
        hse_maker = HSEStaticMaker(input_set_generator=hse_generator)

        # Relaxation flow
        relax_flow = relax_maker.make(structure=structure)
        relax1_job = relax_flow.jobs[0]
        relax2_job = relax_flow.jobs[1]
        relax_dir = relax_flow.output.dir_name

        # Optics flow
        self.optics_maker.static_maker.task_document_kwargs = {"parse_dos": False}
        self.optics_maker.band_structure_maker.task_document_kwargs = {"parse_dos": False}

        optics_flow = self.optics_maker.make(
            structure=relax_flow.output.structure,
            prev_dir=relax_dir,
        )
        static_job = optics_flow.jobs[0]
        optics_job = optics_flow.jobs[1]

        # HSE flow
        static_dir = optics_job.output.dir_name
        hse06_job = hse_maker.make(
            structure=static_job.output.structure,
            prev_dir=static_dir,
        )
        
        
        job_paths = {
            "relax1": relax1_job.output.dir_name,
            "relax2": relax2_job.output.dir_name,
            "static": static_job.output.dir_name,
            "optics": optics_job.output.dir_name,
            "hse06": hse06_job.output.dir_name,
        }



        # ARTATOP jobs
        artatop_jobs = get_artatop_jobs(
            optics_job_output=optics_job.output,
            hse_job_output=hse06_job.output,
            job_paths=job_paths,
            additional_metadata=additional_metadata
        )



        # Final flow with custom output
        return Flow(
            jobs=[relax_flow, optics_flow, hse06_job, artatop_jobs],
        )                                                                                                                                                                                                                                                                                                                                                                                                                       
