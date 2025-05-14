"""Module defining ARTATOP document schemas."""

import gzip
import json
from pathlib import Path
from typing import Any, Optional, Union, List

# TODO: remove this kludge when monty is fixed
from monty.os.path import zpath as monty_zpath
from pydantic import BaseModel, Field
from atomate2 import SETTINGS, __version__
import numpy as np
from emmet.core.structure import StructureMetadata
from monty.dev import requires
from monty.json import MontyDecoder, jsanitize

from pymatgen.io.vasp import Vasprun
from pymatgen.core import Structure
from typing_extensions import Self

from atomate2.common.utils import (
    parse_additional_json,
    parse_custodian,
    parse_transformations,
)


from atomate2.utils.datetime import datetime_str

try:
    import ijson
except ImportError:
    ijson = None


class LinearOpticalResponse(BaseModel):
    """Represents the linear optical response data."""

    energy: float = Field(..., description="Energy value in eV.")
    real_part: float = Field(..., description="Real part of the dielectric function.")
    imaginary_part: float = Field(
        ..., description="Imaginary part of the dielectric function."
    )
    absorption_coefficient: Optional[float] = Field(
        None, description="Absorption coefficient at the given energy."
    )
    refractive_index: Optional[float] = Field(
        None, description="Refractive index at the given energy."
    )
    extinction_coefficient: Optional[float] = Field(
        None, description="Extinction coefficient at the given energy."
    )


class NonlinearOpticalResponse(BaseModel):
    """Represents the nonlinear optical response data."""

    energy: float = Field(..., description="Energy value in eV.")
    total_real_part: float = Field(
        ..., description="Total real part of the second harmonic generation response."
    )
    total_imaginary_part: float = Field(
        ...,
        description="Total imaginary part of the second harmonic generation response.",
    )
    contributions: Optional[dict[str, Any]] = Field(
        None, description="Detailed contributions to the nonlinear optical response."
    )   
    
class AtomicContributions(BaseModel):
    atom: str
    orbital_contributions: Optional[dict] = None
    valence_contributions: Optional[dict] = None
    conduction_contributions: Optional[dict] = None
    total_contribution: Optional[float] = None

class DTensorValues(BaseModel):
    energy: float
    components: dict[str, float]

class DeffValues(BaseModel):
    energy: float
    deff: float
class OrbitalContribution(BaseModel):
    orbital: str  # e.g., "s", "p", or "d"
    valence: float
    conduction: float 

class BirefringenceValues(BaseModel):
    energy: float
    delta_n: float
    
class SHGSummaryEntry(BaseModel):
    atom_type: str
    num_atoms: int
    ind: float
    total: float
    vb: float
    cb: float
    vb_s: float
    vb_p: float
    vb_d: float
    cb_s: float
    cb_p: float
    cb_d: float
    tot_s: float
    tot_p: float
    tot_d: float

class DPmVEntry(BaseModel):
    energy: float
    value: float
    label: str  # "d-PmV" or "dshg-PmV"


class ArtatopInputModel(BaseModel):
    """Definition of input settings for the ARTATOP computation."""

    calc_type: str = Field(
        ..., description="Type of calculation (e.g., linear, nonlinear, ART)."
    )
    input_file: str = Field(..., description="Path to the ARTATOP input file.")
    output_dir: str = Field(..., description="Directory for storing output files.")
    custom_components: Optional[dict[str, Any]] = Field(
        None, description="Custom components used for ART calculations."
    )
    task: str = Field(..., description="Specific ARTATOP task or mode to execute.")

    @classmethod
    def from_file(cls, filename: str) -> "ArtatopInputModel":
        # Just use filename to guess the type
        fname = Path(filename).name.lower()

        if "lin" in fname:
            calc_type = "linear"
        elif "nlin" in fname or "nonlin" in fname:
            calc_type = "nonlinear"
        elif "art" in fname:
            calc_type = "art"
        else:
            raise ValueError(f"Could not detect calc_type from filename: {filename}")

        # Default task and other metadata
        task = "default_task"  # You can later extract this from file if needed
        input_file = str(filename)
        output_dir = str(Path(filename).parent)
        custom_components = None  # Set if needed in the future

        return cls(
            calc_type=calc_type,
            input_file=input_file,
            output_dir=output_dir,
            task=task,
            custom_components=custom_components,
        )

class ArtatopOutputModel(BaseModel):
    """Definition of output results from the ARTATOP computation."""

    dir_name: str = Field(..., description="Directory containing ARTATOP outputs.")
    
    structure: Optional[Structure] = None
    
    # Base (non-spin-resolved)
    linear_response: list[LinearOpticalResponse]
    nonlinear_response: list[NonlinearOpticalResponse]
    d_tensor: Optional[list[DTensorValues]]
    deff_values: Optional[list[DeffValues]]
    birefringence: Optional[list[BirefringenceValues]]

    # UV region (new)
    linear_response_uv: Optional[list[LinearOpticalResponse]] = None
    nonlinear_response_uv: Optional[list[NonlinearOpticalResponse]] = None
    d_tensor_uv: Optional[list[DTensorValues]] = None
    deff_values_uv: Optional[list[DeffValues]] = None
    birefringence_uv: Optional[list[BirefringenceValues]] = None

    # IR region (new)
    linear_response_ir: Optional[list[LinearOpticalResponse]] = None
    nonlinear_response_ir: Optional[list[NonlinearOpticalResponse]] = None
    d_tensor_ir: Optional[list[DTensorValues]] = None
    deff_values_ir: Optional[list[DeffValues]] = None
    birefringence_ir: Optional[list[BirefringenceValues]] = None

    # Spin-resolved (optional)
    linear_response_up: Optional[list[LinearOpticalResponse]] = None
    linear_response_down: Optional[list[LinearOpticalResponse]] = None
    d_tensor_up: Optional[list[DTensorValues]] = None
    d_tensor_down: Optional[list[DTensorValues]] = None
    deff_values_up: Optional[list[DeffValues]] = None
    deff_values_down: Optional[list[DeffValues]] = None
    birefringence_up: Optional[list[BirefringenceValues]] = None
    birefringence_down: Optional[list[BirefringenceValues]] = None
    
    atomic_contributions: Optional[List[AtomicContributions]] = None
    
    shg_summary: Optional[list[SHGSummaryEntry]] = None
    d_pmV: Optional[list[DPmVEntry]] = None
    dshg_pmV: Optional[list[DPmVEntry]] = None
    
    chemical_formula: Optional[str] = None
    space_group: Optional[str] = None
    point_group: Optional[str] = None
    n_atoms: Optional[int] = None
    
    ori_a: Optional[float] = None
    ori_b: Optional[float] = None
    ori_c: Optional[float] = None
    ori_alpha: Optional[float] = None
    ori_beta: Optional[float] = None
    ori_gamma: Optional[float] = None
    original_structure: Optional[dict] = None
    
    relax_a: Optional[float] = None
    relax_b: Optional[float] = None
    relax_c: Optional[float] = None
    relax_alpha: Optional[float] = None
    relax_beta: Optional[float] = None
    relax_gamma: Optional[float] = None
    relax_volume: Optional[float] = None
    
    v_over_eg_exp_per_atom: Optional[float] = None
    v_over_eg_hse_per_atom: Optional[float] = None
    bandgap_pbe: Optional[float] = None
    bandgap_exp: Optional[float] = None
    bandgap_hse: Optional[float] = None
    scissor_exp: Optional[float] = None
    scissor_hse: Optional[float] = None

    nbands_static: Optional[int] = None
    nbands_optics: Optional[int] = None
    total_energy_static: Optional[float] = None

    kpoints_relax1: Optional[list[int]] = None
    kpoints_relax2: Optional[list[int]] = None
    kpoints_static: Optional[list[int]] = None
    kpoints_optics: Optional[list[int]] = None
    kpoints_hse06: Optional[list[int]] = None
    
    
    ediffg_relax2: Optional[float] = None
    aexx_hse: Optional[float] = None
    
    art_top_component: Optional[str] = None
    art_top_value: Optional[float] = None
    
    @classmethod
    def from_directory(
        cls,
        dir_name: str,
        input_file: str,
        job_paths: Optional[dict] = None,
        pbe_vasprun_file: Optional[Path] = None,
        hse_vasprun_file: Optional[Path] = None,
        additional_metadata: dict = None,
    ) -> "ArtatopOutputModel":    
        from atomate2.artatop.artatop_parser import parse_artatop_outputs, get_artatop_functional_data
        return parse_artatop_outputs(
            dir_name=dir_name,
            input_file=input_file,
            job_paths=job_paths,
            pbe_vasprun_file=pbe_vasprun_file,
            hse_vasprun_file=hse_vasprun_file,
            additional_metadata=additional_metadata,
        )
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Structure: lambda v: v.as_dict()
        }


class ArtatopTaskDocument(StructureMetadata, extra="allow"):
    """Main schema for an ARTATOP task document."""
    structure: Structure = Field(description="The structure used in this task")
    
    dir_name: Union[str, Path] = Field(..., description="Directory containing ARTATOP outputs.")
    
    input_data: ArtatopInputModel = Field(
        ..., description="Input parameters for the ARTATOP computation."
    )
    output_data: ArtatopOutputModel = Field(
        ..., description="Parsed results from the ARTATOP computation."
    )
    
    additional_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional optional metadata related to the task."
    )
    
    builder_meta: dict[str, Union[str, None]] = Field(
        default_factory=dict,
        description="Metadata such as builder source and version info."
    )
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp when this task document was last updated.",
    )
    additional_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for this task."
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: str,
        input_file: str,
        pbe_vasprun_file: Optional[Path] = None,
        hse_vasprun_file: Optional[Path] = None,
        store_additional_json: bool = SETTINGS.ARTATOP_STORE_ADDITIONAL_JSON,
        additional_metadata: dict = None,
        job_paths: Optional[dict] = None,
    ) -> "ArtatopTaskDocument":
        """Parse ARTATOP inputs and outputs, then construct the task document."""
        from pymatgen.io.vasp import Vasprun

        # --- Load ARTATOP input model ---
        input_data = ArtatopInputModel.from_file(input_file)

        # --- Load ARTATOP parsed outputs ---
        output_data = ArtatopOutputModel.from_directory(
            dir_name=dir_name,
            input_file=input_file,
            job_paths=job_paths,
            pbe_vasprun_file=pbe_vasprun_file,
            hse_vasprun_file=hse_vasprun_file,
            additional_metadata=additional_metadata
        )

        if additional_metadata is None:
            additional_metadata = {}
        builder_meta = {"source": "artatop", "version": __version__}

        # --- Optionally save JSON summary ---
        if store_additional_json:
            art_json_path = Path(dir_name) / "artatop_summary.json.gz"
            with gzip.open(art_json_path, "wt", encoding="UTF-8") as file:
                file.write("[")
                blocks = [
                    {"output_data": output_data},
                    {"additional_metadata": additional_metadata},
                    {"builder_meta": builder_meta},
                ]
                for i, block in enumerate(blocks):
                    json.dump(jsanitize(block, strict=True, allow_bson=True), file)
                    if i < len(blocks) - 1:
                        file.write(",")
                file.write("]")
                
        

        # --- Return the full task document ---
        return cls(
            structure=output_data.structure,
            dir_name=str(dir_name),
            input_data=input_data,
            output_data=output_data,
            builder_meta=builder_meta,
            additional_metadata=additional_metadata,
        )
        
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Structure: lambda v: v.as_dict()
        }
        
def read_saved_json(
    filename: str, pymatgen_objs: bool = True, query: str = "structure"
) -> dict[str, Any]:
    r"""
    Read the data from  \*.json.gz files corresponding to query.

    Uses ijson to parse specific keys(memory efficient)

    Parameters
    ----------
    filename: str.
        name of the json file to read
    pymatgen_objs: bool.
        if True will convert structure,coop, cobi, cohp and dos to pymatgen objects
    query: str or None.
        field name to query from the json file. If None, all data will be returned.

    Returns
    -------
    dict
        Returns a dictionary with artatop task json data corresponding to query.
    """
    with gzip.open(filename, "rb") as file:
        artatop_data = {
            field: data
            for obj in ijson.items(file, "item", use_float=True)
            for field, data in obj.items()
            if query is None or query in obj
        }
        if not artatop_data:
            raise ValueError(
                "Please recheck the query argument. "
                f"No data associated to the requested 'query={query}' "
                f"found in the JSON file"
            )
    if pymatgen_objs:
        for query_key, value in artatop_data.items():
            if isinstance(value, dict):
                artatop_data[query_key] = MontyDecoder().process_decoded(value)
            elif "lobsterpy_data" in query_key:
                for field in artatop_data[query_key].__fields__:
                    val = MontyDecoder().process_decoded(
                        getattr(artatop_data[query_key], field)
                    )
                    setattr(artatop_data[query_key], field, val)

    return artatop_data

    def save_to_json(self, filename: str) -> None:
        """Save the task document as a compressed JSON file."""
        with gzip.open(filename, "wt", encoding="UTF-8") as file:
            json.dump(self.dict(), file, indent=4)
