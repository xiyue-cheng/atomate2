"""
Flows for ARTATOP computations.

This module provides a unified workflow for ARTATOP calculations combined
with VASP workflows, including relaxation, optics, and ARTATOP jobs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from pathlib import Path

from jobflow import Flow, Maker
from pymatgen.core import Structure
from atomate2.common.files import copy_files

from atomate2.vasp.flows.core import DoubleRelaxMaker, OpticsMaker
from atomate2.artatop.job.artatop import get_artatop_jobs
from atomate2.vasp.sets.core import NonSCFSetGenerator
from atomate2.artatop.jobs import ARTATOPMaker

try:
    import ijson
except ImportError:
    ijson = None

from atomate2.vasp.jobs.base import BaseVaspMaker


class ArtatopWorkflowMaker(Maker):
    """Workflow to run ARTATOP after a full VASP double relaxation and optics calculation."""

    name: str = "artatop_workflow"
    relax_maker: Maker = DoubleRelaxMaker()
    optics_maker: Maker = OpticsMaker()

    def make(self, structure: Structure, prev_dir: str | Path) -> Flow:
        """Create a full ARTATOP workflow including relaxation, optics, and ARTATOP jobs."""

        # Inject NBANDS × 4 logic into the band structure step of optics
        self.optics_maker.band_structure_maker.input_set_generator = NonSCFSetGenerator(
            optics=True,
            nbands_factor=4.0
        )

        # Step 1: Relaxation
        relax_flow = self.relax_maker.make(structure=structure)
        relax_dir = relax_flow.output.dir_name

        # Step 2: Optics (static + non-scf with LOPTICS)
        optics_flow = self.optics_maker.make(
            structure=relax_flow.output.structure,
            prev_dir=relax_dir,
        )

        # Step 3: ARTATOP
        artatop_jobs = get_artatop_jobs(optics_job_output=optics_flow.output)

        # Step 4: Return the full flow
        return Flow(
            jobs=[relax_flow, optics_flow, artatop_jobs],
            output=artatop_jobs.output
        )
