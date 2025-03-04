"""
Flows for ARTATOP computations.

This module provides a unified workflow for ARTATOP calculations combined
with VASP workflows, including relaxation, optics, and ARTATOP jobs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

from jobflow import Flow, Maker, job
from pymatgen.core import Structure
from atomate2.common.files import copy_files

from atomate2.vasp.flows.core import DoubleRelaxMaker, OpticsMaker
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.artatop.jobs import ARTATOPMaker
from atomate2.artatop.job.artatop import get_artatop_jobs

try:
    import ijson
except ImportError:
    ijson = None

from pymatgen.core import Structure

class ArtatopWorkflowMaker(Maker):
    """Workflow to run ARTATOP after a full VASP double relaxation and optics calculation."""

    name: str = "artatop_workflow"
    relax_maker: Maker = DoubleRelaxMaker()  
    optics_maker: Maker = OpticsMaker()

    def make(self, structure: Structure, prev_dir: str | Path) -> Flow:
        """Create a full ARTATOP workflow including relaxation, optics, and ARTATOP jobs."""

        
        # Step 1: Relaxation Job
        relax_flow = self.relax_maker.make(structure=structure)
        relax_dir = relax_flow.output.dir_name

        # Step 2: Optics Job (after relaxation)
        optics_flow = self.optics_maker.make(
            structure=relax_flow.output.structure,
            prev_dir=relax_flow.output.dir_name,
        )
        print(f"Optics output refernce: {optics_flow.output}")
        artatop_jobs = get_artatop_jobs(None, {"dir_name": optics_flow.output.dir_name)

        # **4. Return the Final Flow**
        return Flow(
            jobs=[relax_flow, optics_flow, artatop_jobs],
            output=artatop_jobs.output
        )
