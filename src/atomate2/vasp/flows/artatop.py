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
from atomate2.vasp.jobs.artatop import (  # Assuming these are implemented as job makers
    get_lin_job,
    get_nlin_job,
    get_art_job,
)

try:
    import ijson
except ImportError:
    ijson = None

from pymatgen.core import Structure


"""
Flows for ARTATOP computations.

This module provides a unified workflow for ARTATOP calculations combined
with VASP workflows, including relaxation, optics, and ARTATOP jobs.
"""


class ArtatopWorkflowMaker(Maker):
    """Unified workflow maker for ARTATOP with VASP workflows."""

    name: str = "artatop_workflow"
    relax_maker: Maker = DoubleRelaxMaker()
    optics_maker: Maker = OpticsMaker()

    def make(self, structure: Structure, prev_dir: str | Path) -> Flow:
        """Create a unified workflow combining relaxation, optics, and ARTATOP."""
        
        # Step 1: Relaxation Job
        relax_flow = self.relax_maker.make(structure=structure)

        # Step 2: Optics Job (after relaxation)
        optics_flow = self.optics_maker.make(
            structure=relax_flow.output.structure,
            prev_dir=relax_flow.output.dir_name,
        )

        # Step 4: Define ARTATOP Jobs (LIN, NLIN, and ART)
        lin_job = get_lin_job(prev_dir=optics_flow.output.dir_name)
        nlin_job = get_nlin_job(prev_dir=optics_flow.output.dir_name)
        art_job = get_art_job(prev_dir=nlin_job.output["nlin_output"])
  

        # Combine all jobs into a single Flow
        return Flow(
            jobs=[
                relax_flow,  # Unpack jobs from relax_flow
                optics_flow,  # Unpack jobs from optics_flow
                lin_job,
                nlin_job,
                art_job,
            ],
            output=art_job.output,
        )
