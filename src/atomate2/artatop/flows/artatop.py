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

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import NonSCFMaker, RelaxMaker, StaticMaker
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.artatop.job.artatop import get_artatop_jobs
from atomate2.vasp.sets.core import NonSCFSetGenerator, StaticSetGenerator
from atomate2.artatop.jobs import ARTATOPMaker

try:
    import ijson
except ImportError:
    ijson = None

from pymatgen.core import Structure
from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class VaspARTATOPMaker(Maker):
    """
    Maker to perform an ARTATOP computation.

    The calculations performed are:

    1. **Double relaxation** (to fully relax the structure).
    2. **Static calculation** (to obtain charge density, necessary for optics).
    3. **Optics calculation** (generates WAVEDER, OPTICS).
    4. **ARTATOP calculation**.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker
        Maker for structure relaxation.
    static_maker : .BaseVaspMaker
        Maker for static calculation.
    optics_maker : .BaseVaspMaker
        Maker for optics calculation (generates WAVEDER, OPTICS).
    artatop_maker : .ARTATOPMaker
        Maker for the ARTATOP calculation.
    delete_waveders : bool
        If true, WAVEDER files will be deleted after the run.
    """

    name: str = "artatop"
    relax_maker: BaseVaspMaker = field(default_factory=lambda: DoubleRelaxMaker())
    static_maker: BaseVaspMaker = field(default_factory=lambda: StaticMaker())
    optics_maker: BaseVaspMaker = field(default_factory=lambda: NonSCFMaker(
        name="optics",
        input_set_generator=NonSCFSetGenerator(optics=True),
    ))
    artatop_maker: ARTATOPMaker = field(default_factory=ARTATOPMaker)
    delete_waveders: bool = True

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Make flow to calculate optical properties with ARTATOP.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure.
        prev_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        """
        jobs = []

        # 1. Double relaxation
        relax_job = self.relax_maker.make(structure, prev_dir=prev_dir)
        jobs.append(relax_job)
        structure = relax_job.output.structure
        relax_dir = relax_job.output.dir_name
        prev_dir = relax_dir

        # 2. Static calculation
        static_job = self.static_maker.make(structure, prev_dir=prev_dir)
        jobs.append(static_job)
        static_dir = static_job.output.dir_name
        prev_dir = static_dir  # Now optics uses this directory

        # 3. Optics calculation (static → optics)
        optics_job = self.optics_maker.make(structure, prev_dir=prev_dir)
        jobs.append(optics_job)
        optics_dir = optics_job.output.dir_name
        optics_uuid = optics_job.output.uuid

        # 4. ARTATOP calculation using optics_dir as wavefunction_dir
        artatop_jobs = get_artatop_jobs(
            artatop_maker=self.artatop_maker,
            optics_dir=optics_dir,
            optics_uuid=optics_uuid,
        )
        jobs.append(artatop_jobs)

        # Delete WAVEDER files after ARTATOP run
        if self.delete_waveders:
            delete_waveders = delete_artatop_waveder(
                dirs=artatop_jobs.output["artatop_dirs"],
                optics_dir=optics_dir,
            )
            jobs.append(delete_waveders)

        return Flow(jobs, output=artatop_jobs.output)
