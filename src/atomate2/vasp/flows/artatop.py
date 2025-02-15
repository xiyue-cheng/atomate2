"""
Flows for ARTATOP computations.

This module provides a unified workflow for ARTATOP calculations combined
with VASP workflows, including relaxation, optics, and ARTATOP jobs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from pymatgen.core import Structure

from atomate2.vasp.flows.core import DoubleRelaxMaker, OpticsMaker
from atomate2.vasp.jobs.artatop import (  # Assuming these are implemented as job makers
    ARTMaker,
    LINMaker,
    NLINMaker,
)

try:
    import ijson
except ImportError:
    ijson = None
if TYPE_CHECKING:
    from pymatgen.core import Structure


class ArtatopWorkflowMaker(Maker):
    """Unified workflow maker for ARTATOP with VASP workflows."""

    name: str = "artatop_workflow"
    relax_maker: Maker = DoubleRelaxMaker()
    optics_maker: Maker = OpticsMaker()
    lin_maker: Maker = LINMaker()
    nlin_maker: Maker = NLINMaker()
    art_maker: Maker = ARTMaker()

    def make(self, structure: Structure) -> Flow:
        """Create a unified workflow combining relaxation, optics, and ARTATOP."""
        # Step 1: Relaxation Flow
        relax_flow = self.relax_maker.make(structure=structure)

        # Step 2: Optics Flow
        optics_flow = self.optics_maker.make(
            structure=relax_flow.output.structure,
            prev_dir=relax_flow.output.dir_name,
        )

        # Step 3: ARTATOP Jobs
        lin_job = self.lin_maker.make(prev_dir=optics_flow.output.dir_name)
        nlin_job = self.nlin_maker.make(prev_dir=optics_flow.output.dir_name)
        art_job = self.art_maker.make(prev_dir=nlin_job.output["nlin_output"])

        # Combine all into a single flow
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
