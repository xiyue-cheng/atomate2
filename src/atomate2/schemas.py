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
class OrbitalContribution(BaseModel):
    orbital: str  # e.g., "s", "p", or "d"
    valence: float
    conduction: float   
class AtomicContributions(BaseModel):
    """Represents atomic contributions to nonlinear optical properties."""

    atom: str = Field(..., description="Atom contributing to the property.")
    orbital_contributions: dict[str, float] = Field(
        ..., description="Contributions by orbitals (s, p, d)."
    )
    total_contribution: float = Field(
        ..., description="Total contribution of the atom."
    )
class DTensorValues(BaseModel):
    energy: float
    components: dict[str, float]

class DeffValues(BaseModel):
    energy: float
    deff: float

class BirefringenceValues(BaseModel):
    energy: float
    delta_n: float
    
class OrbitalContribution(BaseModel):
    orbital: str  # e.g., "s", "p", or "d"
    valence: float
    conduction: float 

class BandEnergyInfo(BaseModel):
    band_index: int
    e_prf: float
    e_max: float
    e_min: float
    
class DshgBandContribution(BaseModel):
    band_index: int
    im_total: float
    re_total: float
    im_vb: float
    re_vb: float
    im_cb: float
    re_cb: float

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

    # Base (non-spin-resolved)
    linear_response: list[LinearOpticalResponse]
    nonlinear_response: list[NonlinearOpticalResponse]
    d_tensor: Optional[list[DTensorValues]]
    deff_values: Optional[list[DeffValues]]
    birefringence: Optional[list[BirefringenceValues]]

    # Spin-resolved (optional)
    linear_response_up: Optional[list[LinearOpticalResponse]] = None
    linear_response_down: Optional[list[LinearOpticalResponse]] = None
    d_tensor_up: Optional[list[DTensorValues]] = None
    d_tensor_down: Optional[list[DTensorValues]] = None
    deff_values_up: Optional[list[DeffValues]] = None
    deff_values_down: Optional[list[DeffValues]] = None
    birefringence_up: Optional[list[BirefringenceValues]] = None
    birefringence_down: Optional[list[BirefringenceValues]] = None

    # Atomic contributions (added later)
    atomic_contributions: Optional[list[AtomicContributions]] = None
    
    band_energy_info: Optional[List[BandEnergyInfo]] = None
    dshg_summary: Optional[List[DshgBandContribution]] = None

    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp when this output was last updated.",
    )

    @classmethod
    def from_directory(cls, dir_name: str) -> "ArtatopOutputModel":
        """Parse ARTATOP outputs and construct the output document."""
        from atomate2.artatop.artatop_parser import (
            parse_linear_response_at_energies,
            parse_nonlinear_response_at_energies,
            parse_orbital_atomic_contributions,
            write_result_art_IND,
        )

        dir_path = Path(dir_name)
        energies = [0.0, 0.65, 1.167]

        lin = parse_linear_response_at_energies(dir_path / "out_lin", energies)
        nlin = parse_nonlinear_response_at_energies(dir_path / "out_nonlin", energies)
        # After ART step: add atomic orbital contributions
        structure = Structure.from_file(Path(dir_name) / "POSCAR")
        atomic_contribs = parse_orbital_atomic_contributions(structure, Path(dir_name) / "out_nonlin")

        # Save .IND file
        write_result_art_IND(atomic_contribs, filename=Path(dir_name) / "result.art_IND")

        return cls(
            dir_name=str(dir_path),


            # Base
            linear_response=lin["base"]["linear"],
            nonlinear_response=nlin["base"]["nonlinear"],
            d_tensor=nlin["base"]["d_tensor"],
            deff_values=nlin["base"]["deff"],
            birefringence=lin["base"]["birefringence"],

            # Spin Up
            linear_response_up=lin["up"]["linear"] if lin.get("up") else None,
            d_tensor_up=nlin["up"]["d_tensor"] if nlin.get("up") else None,
            deff_values_up=nlin["up"]["deff"] if nlin.get("up") else None,
            birefringence_up=lin["up"]["birefringence"] if lin.get("up") else None,

            # Spin Down
            linear_response_down=lin["down"]["linear"] if lin.get("down") else None,
            d_tensor_down=nlin["down"]["d_tensor"] if nlin.get("down") else None,
            deff_values_down=nlin["down"]["deff"] if nlin.get("down") else None,
            birefringence_down=lin["down"]["birefringence"] if lin.get("down") else None,

            atomic_contributions=atomic_contribs,
            band_energy_info=band_energy_info,
            dshg_summary=dshg_data,
        )
    def save_to_json(self, filename: str) -> None:
        """Save the task document as a compressed JSON file."""
        with gzip.open(filename, "wt", encoding="UTF-8") as f:
            json.dump(self.dict(), f, indent=4)


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
        store_additional_json: bool = SETTINGS.ARTATOP_STORE_ADDITIONAL_JSON,
        additional_metadata: dict = None,
    ) -> "ArtatopTaskDocument":
        """Parse ARTATOP inputs and outputs, then construct the task document."""
        from pymatgen.io.vasp import Vasprun
        
        # Load input data
        input_data = ArtatopInputModel.from_file(input_file)

        # Parse outputs
        output_data = ArtatopOutputModel.from_directory(dir_name)
        
        vasprun_path = Path(dir_name) / "vasprun.xml"
        if not vasprun_path.exists():
            raise FileNotFoundError(f"No vasprun.xml found in {dir_name} to load structure.")

        vasprun = Vasprun(str(vasprun_path))
        structure = vasprun.final_structure


        # Construct and return the task document
        return cls(
            structure=structure,
            dir_name=str(dir_name),
            input_data=input_data,
            output_data=output_data,
            additional_metadata=additional_metadata or {},
        )

    def save_to_json(self, filename: str) -> None:
        """Save the task document as a compressed JSON file."""
        with gzip.open(filename, "wt", encoding="UTF-8") as file:
            json.dump(self.dict(), file, indent=4)
