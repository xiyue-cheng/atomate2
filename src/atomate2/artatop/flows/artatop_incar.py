from __future__ import annotations
from typing import Optional
from pathlib import Path

from jobflow import Flow, Maker
from pymatgen.core import Structure
from pymatgen.io.vasp import Vasprun, Incar

from atomate2.vasp.jobs.core import RelaxMaker, HSEStaticMaker, StaticMaker, NonSCFMaker
from atomate2.vasp.sets.core import RelaxSetGenerator, StaticSetGenerator, HSEStaticSetGenerator, NonSCFSetGenerator
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.artatop.job.artatop import get_artatop_jobs


from atomate2.vasp.sets.core import NonSCFSetGenerator
from typing import Optional
from pymatgen.io.vasp.inputs import Incar
from pymatgen.core import Structure


def safe_user_incar(base: dict) -> dict:
    cleaned = {k: v for k, v in base.items() if v is not None}
    for k, v in cleaned.items():
        if isinstance(v, str | float | int):
            continue
        if v is None:
            print(f"[BUG] INCAR key {k} has None value!")
        else:
            print(f"[CHECK] INCAR key {k} has suspicious value: {v}")
    return cleaned

class PureNonSCFGenerator(NonSCFSetGenerator):
    def __init__(self, nbands_factor: float = 4.0, user_incar_settings: Optional[dict] = None, **kwargs):
        self._raw_user_incar = user_incar_settings or {}
        self.k_hse = 0.30  # default, will be updated in get_input_set
        super().__init__(nbands_factor=nbands_factor, user_incar_settings=user_incar_settings, **kwargs)

    def get_input_set(self, structure, prev_dir: Optional[str] = None, **kwargs):
        """
        Override get_input_set to apply our NBANDS → NEDOS/KSPACING logic.
        """
        vis = super().get_input_set(structure, prev_dir=prev_dir, **kwargs)

        # Safely apply NBANDS logic based on prev_dir (vasprun.xml)
        nbands = None
        if prev_dir:
            vasprun_path = Path(prev_dir) / "vasprun.xml"
            if vasprun_path.exists():
                try:
                    vasprun = Vasprun(vasprun_path)
                    nbands = vasprun.parameters.get("NBANDS")
                    if nbands:
                        nbands = int(nbands * self.nbands_factor)
                except Exception as e:
                    print(f"Warning: Could not read vasprun.xml from {prev_dir}: {str(e)}")

        if nbands:
            vis.incar["NBANDS"] = nbands
            if nbands >= 250:
                vis.incar["NEDOS"] = 6001
                vis.incar["KSPACING"] = 0.20
                self.k_hse = 0.38
            elif nbands >= 170:
                vis.incar["NEDOS"] = 4001
                vis.incar["KSPACING"] = 0.16
                self.k_hse = 0.34
            else:
                vis.incar["NEDOS"] = 2001
                vis.incar["KSPACING"] = 0.12
                self.k_hse = 0.30

        return vis


# --- Main Flow ---
class ArtatopWorkflowMaker(Maker):
    name: str = "artatop_workflow"
    relax_maker: Maker = DoubleRelaxMaker()
    hse06_maker: Maker = HSEStaticMaker()

    def make(self, structure: Structure, prev_dir: str | Path, additional_metadata: Optional[dict] = None) -> Flow:
        # --- Relaxation ---
        relax_generator = RelaxSetGenerator(
            user_incar_settings={
                "EDIFFG": -0.02, "LREAL": "Auto", "NPAR": 8, "LORBIT": 10,
                "LAECHG": None, "LASPH": None, "LVTOT": None, "GGA": None,
                "MAGMOM": None, "SIGMA": None, "ALGO": None, "ENAUG": None,
                "LMIXTAU": None, "KSPACING": 0.2, "KGAMMA": True
            },
            user_kpoints_settings= None)
        relax1 = RelaxMaker(input_set_generator=relax_generator)
        relax2 = RelaxMaker(input_set_generator=relax_generator)
        relax_maker = DoubleRelaxMaker(relax_maker1=relax1, relax_maker2=relax2)
        relax_flow = relax_maker.make(structure=structure)

        relax1_job = relax_flow.jobs[0]
        relax2_job = relax_flow.jobs[1]
        relax_dir = relax_flow.output.dir_name

        # --- Static ---
        static_generator = StaticSetGenerator(
            user_incar_settings={
                "LORBIT": 10, "NPAR": 8, "LAECHG": None, "LASPH": None,
                "LVTOT": None, "GGA": None, "MAGMOM": None, "SIGMA": None,
                "ALGO": None, "ENAUG": None, "LMIXTAU": None, "KSPACING": 0.12,
                "KGAMMA": True
            },
            user_kpoints_settings= None)
            
        static_maker = StaticMaker(input_set_generator=static_generator)
        static_job = static_maker.make(structure=relax_flow.output.structure, prev_dir=relax_dir)
        static_dir = static_job.output.dir_name
        

        # --- Optics ---
        optics_generator = PureNonSCFGenerator(
            optics=True,
            nbands_factor=4.0,
            user_incar_settings={
                "CSHIFT": 0.1,
                "LORBIT": 10,
                "NPAR": 1,
                "ISYM": 0,
                "KGAMMA": True, "LAECHG": None, "LASPH": None,
                "LVTOT": None, "GGA": None, "MAGMOM": None, "SIGMA": None,
                "ALGO": None, "ENAUG": None, "LMIXTAU": None              
            },
            user_kpoints_settings= None)

        band_structure_maker = NonSCFMaker(input_set_generator=optics_generator)
        optics_job = band_structure_maker.make(structure=static_job.output.structure, prev_dir=static_dir)
        optics_job.name = "optics"

        # --- HSE ---
        def safe_user_incar(base: dict) -> dict:
            """Remove None values and return cleaned INCAR dict."""
            return {k: v for k, v in base.items() if v is not None}

        kspacing_hse = optics_generator.k_hse

        hse_incar_settings = safe_user_incar({
            "GGA": None, "NPAR": 8, "LAECHG": None, "LASPH": None,
            "LVTOT": None, "MAGMOM": None, "SIGMA": None, "ALGO": "Damped",
            "ENAUG": None, "LMIXTAU": None,
            "KSPACING": kspacing_hse,
            "KGAMMA": True, "LMAXMIX": True
        })

        assert all(v is not None for v in hse_incar_settings.values()), "INCAR contains None!"

        hse_generator = HSEStaticSetGenerator(user_incar_settings=hse_incar_settings)
        hse_maker = HSEStaticMaker(input_set_generator=hse_generator)
        hse06_job = hse_maker.make(structure=static_job.output.structure, prev_dir=static_dir)

        # --- Job Paths ---
        job_paths = {
            "relax1": relax1_job.output.dir_name,
            "relax2": relax2_job.output.dir_name,
            "static": static_job.output.dir_name,
            "optics": optics_job.output.dir_name,
            "hse06": hse06_job.output.dir_name,
        }

        # --- ARTATOP ---
        artatop_jobs = get_artatop_jobs(
            optics_job_output=optics_job.output,
            hse_job_output=hse06_job.output,
            job_paths=job_paths,
            additional_metadata=additional_metadata
        )

        return Flow(
            jobs=[relax_flow, static_job, optics_job, hse06_job, artatop_jobs],
            name="artatop"
        )
