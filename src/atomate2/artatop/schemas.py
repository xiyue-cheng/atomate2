"""Module defining ARTATOP document schemas."""

import gzip
import json
from pathlib import Path
from typing import Any, Optional

# TODO: remove this kludge when monty is fixed
from pydantic import BaseModel, Field

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
    """Represents atomic contributions to nonlinear optical properties."""

    atom: str = Field(..., description="Atom contributing to the property.")
    orbital_contributions: dict[str, float] = Field(
        ..., description="Contributions by orbitals (s, p, d)."
    )
    total_contribution: float = Field(
        ..., description="Total contribution of the atom."
    )


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
        """Load input settings from an ARTATOP input file."""
        with open(filename) as f:
            input_data = json.load(f)
        return cls(**input_data)

    def to_file(self, filename: str) -> None:
        """Save input settings to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.dict(), f, indent=4)


class ArtatopOutputModel(BaseModel):
    """Definition of output results from the ARTATOP computation."""

    dir_name: str = Field(..., description="Directory containing ARTATOP outputs.")
    linear_response: list[LinearOpticalResponse] = Field(
        ..., description="Parsed linear optical response data."
    )
    nonlinear_response: list[NonlinearOpticalResponse] = Field(
        ..., description="Parsed nonlinear optical response data."
    )
    atomic_contributions: Optional[list[AtomicContributions]] = Field(
        None, description="Atomic contributions to nonlinear optical properties."
    )
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp when this output was last updated.",
    )

    @classmethod
    def from_directory(cls, dir_name: str) -> "ArtatopOutputModel":
        """Parse ARTATOP outputs and construct the output document."""
        from atomate2.artatop.parsers import (
            parse_atomic_contributions,
            parse_linear_response,
            parse_nonlinear_response,
        )

        dir_path = Path(dir_name)

        # Parse outputs
        linear_response = parse_linear_response(dir_path / "out_lin")
        nonlinear_response = parse_nonlinear_response(dir_path / "out_nonlin")
        atomic_contributions = (
            parse_atomic_contributions(dir_path / "out_nonlin")
            if (dir_path / "out_nonlin" / "arp_nonlin.txt").exists()
            else None
        )

        return cls(
            dir_name=str(dir_name),
            linear_response=linear_response,
            nonlinear_response=nonlinear_response,
            atomic_contributions=atomic_contributions,
        )

    def save_to_json(self, filename: str) -> None:
        """Save the task document as a compressed JSON file."""
        with gzip.open(filename, "wt", encoding="UTF-8") as f:
            json.dump(self.dict(), f, indent=4)


class ArtatopTaskDocument(BaseModel):
    """Main schema for an ARTATOP task document."""

    dir_name: str = Field(..., description="Directory containing ARTATOP outputs.")
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
        additional_metadata: dict = None,
    ) -> "ArtatopTaskDocument":
        """Parse ARTATOP inputs and outputs, then construct the task document."""
        # Load input data
        input_data = ArtatopInputModel.from_file(input_file)

        # Parse outputs
        output_data = ArtatopOutputModel.from_directory(dir_name)

        # Construct and return the task document
        return cls(
            dir_name=str(dir_name),
            input_data=input_data,
            output_data=output_data,
            additional_metadata=additional_metadata or {},
        )

    def save_to_json(self, filename: str) -> None:
        """Save the task document as a compressed JSON file."""
        with gzip.open(filename, "wt", encoding="UTF-8") as file:
            json.dump(self.dict(), file, indent=4)
