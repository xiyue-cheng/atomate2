"""
Flows for ARTATOP computations.

This module provides a unified workflow for ARTATOP calculations combined
with VASP workflows, including relaxation, optics, and ARTATOP jobs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

from jobflow import Flow, Maker
from pymatgen.core import Structure
from atomate2.common.files import copy_files

from atomate2.vasp.flows.core import DoubleRelaxMaker, OpticsMaker
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.artatop.jobs import ARTATOPMaker
from atomate2.artatop.job.artatop import (  # Assuming these are implemented as job makers
    ArtatopOpticsMaker,
    get_lin_jobs,
    get_nlin_jobs,
    get_art_jobs
)

try:
    import ijson
except ImportError:
    ijson = None

from pymatgen.core import Structure


if TYPE_CHECKING:
    from pymatgen.core import Structure

class ArtatopWorkflowMaker(Maker):
    """Workflow to run ARTATOP after a full VASP calculation sequence."""

    name: str = "artatop_workflow"
    relax_maker: Maker = DoubleRelaxMaker()
    static_maker: Maker = StaticMaker()
    optics_maker: Maker = ArtatopOpticsMaker()
    artatop_maker: ARTATOPMaker = ARTATOPMaker()

    def make(self, structure: Structure, prev_dir: str | Path) -> Flow:
        """Create a full ARTATOP workflow including VASP calculations."""

        relax_flow = self.relax_maker.make(structure=structure)

        static_job = self.static_maker.make(
            structure=relax_flow.output.structure,
            prev_dir=relax_flow.output.dir_name,
        )

        # **Run ARTATOP Optics Job**
        optics_job = self.optics_maker.make(
            structure=static_job.output.structure,
            prev_dir=static_job.output.dir_name,
        )

        optics_dir = optics_job.output.dir_name

        # **Run ARTATOP Jobs**
        lin_jobs = get_lin_jobs(self.artatop_maker, optics_dir)
        nlin_jobs = get_nlin_jobs(self.artatop_maker, optics_dir)
        art_jobs = get_art_jobs(self.artatop_maker, optics_dir)

        return Flow(jobs=[relax_flow, static_job, optics_job, lin_jobs, nlin_jobs, art_jobs], output=art_jobs.output)
