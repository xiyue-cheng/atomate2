from __future__ import annotations
from typing import Optional, Union
from pathlib import Path

from jobflow import Flow, Maker, job
from pymatgen.core import Structure

from pymatgen.io.vasp import VaspInput, Vasprun
from pymatgen.io.vasp.inputs import Kpoints

from atomate2.vasp.jobs.core import RelaxMaker, HSEStaticMaker, StaticMaker, NonSCFMaker
from atomate2.vasp.sets.core import RelaxSetGenerator, StaticSetGenerator, HSEStaticSetGenerator, NonSCFSetGenerator
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.artatop.job.artatop import get_artatop_jobs
from atomate2.vasp.files import write_vasp_input_set

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.artatop.job.artatop import get_artatop_jobs

from pathlib import Path
from pymatgen.io.vasp.sets import Kpoints
from pymatgen.io.vasp.outputs import Vasprun
from atomate2.vasp.sets.core import NonSCFSetGenerator
from atomate2.vasp.files import write_vasp_input_set
from typing import Optional



class NON_SCF_GENERATOR(NonSCFSetGenerator):
    def __init__(self, nbands_factor: float = 4.0, user_incar_settings: Optional[dict] = None, **kwargs):
        self._raw_user_incar = user_incar_settings or {}
        super().__init__(user_incar_settings=user_incar_settings, **kwargs)
        self.nbands_factor = nbands_factor

    def get_input_set(self, structure, prev_dir: Optional[str] = None, **kwargs):
        vis = super().get_input_set(structure, prev_dir=prev_dir, **kwargs)
        
        # NBANDS adjustment logic
        nbands = None
        if prev_dir:
            try:
                vr = Vasprun(Path(prev_dir)/"vasprun.xml")
                nbands = int(vr.parameters["NBANDS"] * self.nbands_factor)
            except Exception:
                pass

        if nbands:
            vis.incar["NBANDS"] = nbands
            if nbands >= 250:
                vis.incar["NEDOS"] = 6001
                vis.incar["KSPACING"] = 0.20
            elif nbands >= 170:
                vis.incar["NEDOS"] = 4001
                vis.incar["KSPACING"] = 0.16
            else:
                vis.incar["NEDOS"] = 2001
                vis.incar["KSPACING"] = 0.12

        return VaspInput(incar=vis.incar, poscar=vis.poscar, potcar=vis.potcar, kpoints=None)




class HSE_GENERATOR(HSEStaticSetGenerator):
    def __init__(self, nbands_factor: float = 4.0, user_incar_settings: Optional[dict] = None, **kwargs):
        self._raw_user_incar = user_incar_settings or {}
        super().__init__(user_incar_settings=user_incar_settings, **kwargs)
        self.nbands_factor = nbands_factor
    
        
    def get_input_set(self, structure, prev_dir: Optional[str] = None, **kwargs):
        vis = super().get_input_set(structure, prev_dir=prev_dir, **kwargs)
        

        if prev_dir:
            vasprun_path = Path(prev_dir) / "vasprun.xml"
            if vasprun_path.exists():
                try:
                    vasprun = Vasprun(vasprun_path)
                    nbands = vasprun.parameters.get("NBANDS")
                    if nbands:
                        nbands = int(nbands * self.nbands_factor)
                        if nbands >= 250:
                            vis.incar["KSPACING"] = 0.38
                        elif nbands >= 170:
                            vis.incar["KSPACING"] = 0.34
                        else:
                            vis.incar["KSPACING"] = 0.30
                except Exception as e:
                    print(f"Warning reading vasprun.xml: {e}")
        return VaspInput(incar=vis.incar, poscar=vis.poscar, potcar=vis.potcar, kpoints=None)

# --- Main Flow ---
class ArtatopWorkflowMaker(Maker):
    name: str = "artatop_workflow"
    relax_maker: Maker = DoubleRelaxMaker()

    def make(self, structure: Structure, prev_dir: Union[str, Path], additional_metadata: Optional[dict] = None) -> Flow:
        # --- Relaxation ---
        relax_generator = RelaxSetGenerator(
            user_incar_settings={
                "EDIFFG": -0.02, "NPAR": 8, "LORBIT": 10, "LREAL": "Auto", "KSPACING": 0.2, "KGAMMA": True,
                "LAECHG": None, "LASPH": None, "LVTOT": None, "GGA": None, "LMIXTAU": None,  "ISPIN": None,
                "MAGMOM": None, "SIGMA": None, "ALGO": None, "ENAUG": None,   
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
                "LORBIT": 10, "NPAR": 8,  "NELM": 60, "KGAMMA": True, "KSPACING": 0.12, "LREAL": "Auto",
                "LVTOT": None, "GGA": None, "MAGMOM": None, "SIGMA": None, "ISPIN": None, "LAECHG": None, "LASPH": None,
                "ALGO": None, "ENAUG": None, "LMIXTAU": None,  
            },
            user_kpoints_settings= None)
            
        static_maker = StaticMaker(input_set_generator=static_generator,
        task_document_kwargs={"parse_dos": False}
        )
        static_job = static_maker.make(structure=relax_flow.output.structure, prev_dir=relax_dir)
        static_dir = static_job.output.dir_name
        

        # --- Optics ---
        optics_generator = NON_SCF_GENERATOR(
            nbands_factor=4.0,
            user_incar_settings={
                "CSHIFT": 0.1, "LORBIT": 10, "NPAR": 1, "NELM": 60, "KGAMMA": True, "LREAL": "Auto",
                "ISYM": None, "LAECHG": None, "LASPH": None, "LVTOT": None, "GGA": None, "MAGMOM": None,
                "LOPTICS": True, "ISPIN": None, 
                "SIGMA": None, "ALGO": None, "ENAUG": None, "LMIXTAU": None, "LMAXMIX": None,
            },
            user_kpoints_settings= None)

        optics_job = NonSCFMaker(input_set_generator=optics_generator,
        task_document_kwargs={"parse_dos": False}).make(
            structure=static_job.output.structure,
            prev_dir=static_job.output.dir_name,
        )      
        optics_job.name = "optics"
               
        hse_generator = HSE_GENERATOR(
            user_incar_settings={
                "NPAR": 8,  "GGA": None, "ISMEAR": 0, "LAECHG": None, "LASPH": True, "ENAUG": None, "LMIXTAU": None, "ISYM": None, "ISPIN": None,
                "LREAL": "Auto", "LORBIT": 10, "KGAMMA": True, "NELM": 60,
                "LVTOT": None, "MAGMOM": None, "LMAXMIX": None, "LDAU": None,
                "LCHARG": False, "ALGO": "Damped",  "TIME": 0.4, "SIGMA": 0.01, "LHFCALC": True, "HFSCREEN": 0.2, "PRECFOCK": "Normal",
            },
            user_kpoints_settings=None)
            
        hse_job = HSEStaticMaker(input_set_generator=hse_generator).make(
            structure=static_job.output.structure,
            prev_dir=static_job.output.dir_name
        )


        # --- Job Paths ---
        job_paths = {
            "relax1": relax1_job.output.dir_name,
            "relax2": relax2_job.output.dir_name,
            "static": static_job.output.dir_name,
            "optics": optics_job.output.dir_name,
            "hse06": hse_job.output.dir_name,
        }

        # --- ARTATOP ---
        artatop_jobs = get_artatop_jobs(
            optics_job_output=optics_job.output,
            hse_job_output=hse_job.output,
            job_paths=job_paths,
            additional_metadata=additional_metadata
        )

        return Flow(
            jobs=[relax_flow, static_job, optics_job, hse_job, artatop_jobs],
            name="artatop"
        )
