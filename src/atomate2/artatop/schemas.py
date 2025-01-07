from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# RELAXATION OUTPUT SCHEMA
# ---------------------------------------------------------
class RelaxationOutput(BaseModel):
    final_structure: dict = Field(
        ..., description="Relaxed structure in dictionary format."
    )
    energy: float = Field(..., description="Final energy after relaxation.")
    forces: list[list[float]] = Field(..., description="Atomic forces on each atom.")
    stress: list[float] = Field(..., description="Stress tensor components.")


# ---------------------------------------------------------
# OPTICS OUTPUT SCHEMA
# ---------------------------------------------------------
class OpticsOutput(BaseModel):
    dielectric_function: dict[str, list[float]] = Field(
        ..., description="Real and imaginary parts of the dielectric function."
    )
    absorption_coefficient: list[float] = Field(
        ..., description="Absorption coefficient as a function of energy."
    )


# ---------------------------------------------------------
# LINEAR OPTICAL PROPERTIES SCHEMA
# ---------------------------------------------------------
class LinearOpticalData(BaseModel):
    energy: float = Field(..., description="Energy value in eV.")
    properties: dict[str, float] = Field(
        ...,
        description="Optical properties at the given energy (e.g., Re(eps), Im(eps)).",
    )


class LinearOpticalFile(BaseModel):
    tensor_component: str = Field(
        ..., description="Tensor component (e.g., 'XX', 'YY')."
    )
    metadata: dict[str, float] = Field(
        ...,
        description="Metadata for the calculation (e.g., broadening, scissors shift).",
    )
    data: list[LinearOpticalData] = Field(
        ..., description="List of optical property data points."
    )


# ---------------------------------------------------------
# NONLINEAR OPTICAL PROPERTIES SCHEMA
# ---------------------------------------------------------
class NonlinearOpticalData(BaseModel):
    energy: float = Field(..., description="Energy value in eV.")
    total_values: Optional[dict[str, float]] = Field(
        None, description="Total nonlinear optical values (e.g., Tot-Im Chi(-2w,w,w))."
    )
    contributions: Optional[dict[str, float]] = Field(
        None,
        description="Contribution-specific nonlinear optical properties (e.g., Re Inter(2w)).",
    )


class NonlinearOpticalFile(BaseModel):
    tensor_component: str = Field(..., description="Tensor component (e.g., 'YYY').")
    metadata: dict[str, float] = Field(
        ...,
        description="Metadata for the calculation (e.g., broadening, scissors shift).",
    )
    total_data: list[NonlinearOpticalData] = Field(
        ..., description="Total nonlinear optical property data."
    )
    contribution_data: list[NonlinearOpticalData] = Field(
        ..., description="Contribution-specific nonlinear optical property data."
    )


# ---------------------------------------------------------
# ATOMIC CONTRIBUTIONS SCHEMA
# ---------------------------------------------------------
class AtomicContributionData(BaseModel):
    energy: float = Field(..., description="Energy value in eV.")
    contributions: list[float] = Field(
        ..., description="List of atomic contributions for the major component."
    )


class AtomicContributionFile(BaseModel):
    major_component: str = Field(
        ..., description="Tensor component with the highest magnitude (e.g., 'YYY')."
    )
    data: list[AtomicContributionData] = Field(
        ..., description="Atomic contributions data for the major component."
    )


# ---------------------------------------------------------
# ARTATOP TASK DOCUMENT SCHEMA
# ---------------------------------------------------------
class ArtatopTaskDocument(BaseModel):
    dir_name: str = Field(..., description="Directory containing ARTATOP outputs.")
    relaxation_output: Optional[RelaxationOutput] = Field(
        None, description="Parsed relaxation output data."
    )
    optics_output: Optional[OpticsOutput] = Field(
        None, description="Parsed optics output data."
    )
    linear_response: list[LinearOpticalFile] = Field(
        [], description="Parsed linear optical response data."
    )
    nonlinear_response: list[NonlinearOpticalFile] = Field(
        [], description="Parsed nonlinear optical response data."
    )
    atomic_contributions: Optional[AtomicContributionFile] = Field(
        None, description="Atomic contributions data for the major component."
    )

    @classmethod
    def from_directory(cls, dir_name: str) -> "ArtatopTaskDocument":
        """
        Parses all outputs from the workflow and constructs the task document.

        Parameters
        ----------
        dir_name : str
            Path to the ARTATOP output directory.

        Returns
        -------
        ArtatopTaskDocument
            The parsed ARTATOP task document.
        """
        from artatop_parser import (
            parse_atomic_contributions,
            parse_optics_output,
            parse_out_lin,
            parse_out_nonlin,
            parse_relaxation_output,
        )

        # Parse relaxation
        relaxation_output = (
            parse_relaxation_output(f"{dir_name}/relaxation")
            if Path(f"{dir_name}/relaxation").exists()
            else None
        )

        # Parse optics
        optics_output = (
            parse_optics_output(f"{dir_name}/optics")
            if Path(f"{dir_name}/optics").exists()
            else None
        )

        # Parse linear optical files
        linear_response = parse_out_lin(f"{dir_name}/out_lin")

        # Parse nonlinear optical files
        nonlinear_response = parse_out_nonlin(f"{dir_name}/out_nonlin")

        # Parse atomic contributions
        atomic_contributions = parse_atomic_contributions(f"{dir_name}/out_nonlin")

        return cls(
            dir_name=dir_name,
            relaxation_output=relaxation_output,
            optics_output=optics_output,
            linear_response=linear_response,
            nonlinear_response=nonlinear_response,
            atomic_contributions=atomic_contributions,
        )
