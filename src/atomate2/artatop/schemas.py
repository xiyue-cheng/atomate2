"""Module defining ARTATOP document schemas."""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


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


class ArtatopTaskDocument(BaseModel):
    """Main schema for an ARTATOP task document."""

    dir_name: str = Field(..., description="Directory containing ARTATOP outputs.")
    linear_response: list["LinearOpticalResponse"] = Field(
        ..., description="Parsed linear optical response data."
    )
    nonlinear_response: list["NonlinearOpticalResponse"] = Field(
        ..., description="Parsed nonlinear optical response data."
    )
    atomic_contributions: Optional[list["AtomicContributions"]] = Field(
        None, description="Parsed atomic contributions data."
    )

    @classmethod
    def from_directory(cls, dir_name: str) -> "ArtatopTaskDocument":
        """Parse ARTATOP outputs and construct the task document."""
        from atomate2.artatop.files import validate_vasp_and_artatop_outputs
        from atomate2.artatop.parsers import (
            parse_atomic_contributions,
            parse_linear_response,
            parse_nonlinear_response,
        )

        # Convert dir_name to a Path object
        dir_path = Path(dir_name)

        # Validate required files and directories
        validate_vasp_and_artatop_outputs(dir_path)

        # Parse data from the specified directories
        linear_response = parse_linear_response(dir_path / "out_lin")
        nonlinear_response = parse_nonlinear_response(dir_path / "out_nonlin")
        atomic_contributions = (
            parse_atomic_contributions(dir_path / "out_nonlin")
            if (dir_path / "out_nonlin" / "arp_nonlin.txt").exists()
            else None
        )

        # Construct the task document
        return cls(
            dir_name=str(dir_name),
            linear_response=linear_response,
            nonlinear_response=nonlinear_response,
            atomic_contributions=atomic_contributions,
        )
