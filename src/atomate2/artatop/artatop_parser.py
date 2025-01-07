import logging
from pathlib import Path

from pymatgen.io.vasp import Outcar, Vasprun

from atomate2.artatop.schemas import (
    AtomicContributionData,
    AtomicContributionFile,
    LinearOpticalData,
    LinearOpticalFile,
    NonlinearOpticalData,
    NonlinearOpticalFile,
    OpticsOutput,
    RelaxationOutput,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# VALIDATION FUNCTIONS
# ---------------------------------------------------------

DEFAULT_RELAXATION_FILES = ["OUTCAR", "CONTCAR"]
DEFAULT_OPTICS_FILES = ["vasprun.xml"]
DEFAULT_ARTATOP_OUTPUT_FILES = ["re_lin", "re_nlin", "re_art"]
DEFAULT_ARTATOP_OUTPUT_DIRS = ["out_lin", "out_nonlin"]


def validate_required_files(required_files: list[str], directory: Path) -> bool:
    missing_files = [file for file in required_files if not (directory / file).exists()]
    if missing_files:
        logger.error(
            f"Missing required files: {', '.join(missing_files)} in {directory}"
        )
        return False
    return True


def validate_required_dirs(required_dirs: list[str], base_directory: Path) -> bool:
    missing_dirs = [
        dir_name
        for dir_name in required_dirs
        if not (base_directory / dir_name).is_dir()
    ]
    if missing_dirs:
        logger.error(
            f"Missing required directories: {', '.join(missing_dirs)} in {base_directory}"
        )
        return False
    return True


def validate_relaxation_files(directory: Path) -> bool:
    logger.info("Validating relaxation files...")
    return validate_required_files(DEFAULT_RELAXATION_FILES, directory)


def validate_optics_files(directory: Path) -> bool:
    logger.info("Validating optics files...")
    return validate_required_files(DEFAULT_OPTICS_FILES, directory)


def validate_artatop_outputs(directory: Path) -> bool:
    logger.info("Validating ARTATOP outputs...")
    files_valid = validate_required_files(DEFAULT_ARTATOP_OUTPUT_FILES, directory)
    dirs_valid = validate_required_dirs(DEFAULT_ARTATOP_OUTPUT_DIRS, directory)
    return files_valid and dirs_valid


# ---------------------------------------------------------
# PARSING FUNCTIONS
# ---------------------------------------------------------


# Relaxation Parsing
def parse_relaxation_output(output_dir: Path) -> RelaxationOutput:
    if not validate_relaxation_files(output_dir):
        raise ValueError("Missing required relaxation files.")

    outcar = Outcar(output_dir / "OUTCAR")
    structure = Structure.from_file(output_dir / "CONTCAR")

    return RelaxationOutput(
        final_structure=structure.as_dict(),
        energy=outcar.final_energy,
        forces=outcar.forces,
        stress=outcar.stress_tensor,
    )


# Optics Parsing
def parse_optics_output(output_dir: Path) -> OpticsOutput:
    if not validate_optics_files(output_dir):
        raise ValueError("Missing required optics files.")

    vasprun = Vasprun(output_dir / "vasprun.xml", parse_potcar_file=False)

    dielectric_data = vasprun.dielectric
    dielectric_function = {
        "real": dielectric_data[1],
        "imaginary": dielectric_data[2],
    }

    return OpticsOutput(
        dielectric_function=dielectric_function,
        absorption_coefficient=vasprun.optical_absorption_coeff,
    )


# Linear Optical Parsing
def parse_lin_file(file_path: Path) -> LinearOpticalFile:
    with open(file_path) as f:
        lines = f.readlines()

    metadata = {}
    data = []
    properties = []

    for line in lines:
        if line.startswith("#"):
            if "broadening" in line:
                metadata["broadening"] = float(
                    line.split(":")[1].strip().replace("eV", "")
                )
            elif "scissors shift" in line:
                metadata["scissors_shift"] = float(
                    line.split(":")[1].strip().replace("eV", "")
                )
            elif "energy window" in line:
                metadata["energy_window"] = float(
                    line.split(":")[1].strip().replace("eV", "")
                )
            elif "Energy(eV)" in line:
                properties = line.split()[1:]
        elif line.strip():
            parts = line.split()
            energy = float(parts[0])
            values = {prop: float(parts[i + 1]) for i, prop in enumerate(properties)}
            data.append(LinearOpticalData(energy=energy, properties=values))

    tensor_component = file_path.stem.split("_")[-1].upper()
    return LinearOpticalFile(
        tensor_component=tensor_component, metadata=metadata, data=data
    )


def parse_out_lin(directory: Path) -> list[LinearOpticalFile]:
    if not validate_required_dirs(["out_lin"], directory):
        raise ValueError("Missing 'out_lin' directory for linear optical files.")

    lin_files = []
    for file in directory.glob("out_lin/lin_*.dat"):
        lin_files.append(parse_lin_file(file))
    return lin_files


# Nonlinear Optical Parsing
def parse_nonlin_file(file_path: Path) -> NonlinearOpticalFile:
    with open(file_path) as f:
        lines = f.readlines()

    metadata = {}
    total_data = []
    contribution_data = []
    current_section = None

    for line in lines:
        if line.startswith("#"):
            if "broadening" in line:
                metadata["broadening"] = float(
                    line.split(":")[1].strip().replace("eV", "")
                )
            elif "scissors shift" in line:
                metadata["scissors_shift"] = float(
                    line.split(":")[1].strip().replace("eV", "")
                )
            elif "energy window" in line:
                metadata["energy_window"] = float(
                    line.split(":")[1].strip().replace("eV", "")
                )
        elif "Energy" in line:
            if "Tot-Im Chi(-2w,w,w)" in line:
                current_section = "total"
                headers = line.split()[1:]
            elif "Re Inter(2w)" in line:
                current_section = "contributions"
                headers = line.split()[1:]
        elif line.strip() and current_section:
            parts = line.split()
            energy = float(parts[0])
            values = {headers[i]: float(parts[i + 1]) for i in range(len(headers))}
            if current_section == "total":
                total_data.append(
                    NonlinearOpticalData(energy=energy, total_values=values)
                )
            elif current_section == "contributions":
                contribution_data.append(
                    NonlinearOpticalData(energy=energy, contributions=values)
                )

    tensor_component = file_path.stem.split("_")[-1].lower()
    return NonlinearOpticalFile(
        tensor_component=tensor_component,
        metadata=metadata,
        total_data=total_data,
        contribution_data=contribution_data,
    )


def parse_out_nonlin(directory: Path) -> list[NonlinearOpticalFile]:
    if not validate_required_dirs(["out_nonlin"], directory):
        raise ValueError("Missing 'out_nonlin' directory for nonlinear optical files.")

    nonlin_files = []
    for file in directory.glob("out_nonlin/nonlin_*.txt"):
        nonlin_files.append(parse_nonlin_file(file))
    return nonlin_files


# Atomic Contributions Parsing
def parse_result_re(file_path: Path) -> str:
    major_component = None
    max_value = -float("inf")

    with open(file_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                component = parts[0]
                magnitude = float(parts[1])
                if magnitude > max_value:
                    major_component = component
                    max_value = magnitude
    return major_component


def parse_atomic_contribution(file_path: Path) -> list[AtomicContributionData]:
    atomic_contributions = []

    with open(file_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                energy = float(parts[0])
                contributions = [float(value) for value in parts[1:]]
                atomic_contributions.append(
                    AtomicContributionData(energy=energy, contributions=contributions)
                )

    return atomic_contributions


def parse_atomic_contributions(out_nonlin_dir: Path) -> AtomicContributionFile:
    result_re_path = out_nonlin_dir / "result.re"
    if not result_re_path.exists():
        raise ValueError("Missing 'result.re' file for atomic contributions.")

    major_component = parse_result_re(result_re_path)
    atomic_file_path = out_nonlin_dir / f"re_major_component_{major_component}.dat"
    if not atomic_file_path.exists():
        raise ValueError(f"Missing file: {atomic_file_path}")

    atomic_data = parse_atomic_contribution(atomic_file_path)
    return AtomicContributionFile(major_component=major_component, data=atomic_data)


# ---------------------------------------------------------
# CONSOLIDATED PARSING FUNCTION
# ---------------------------------------------------------


def parse_artatop_outputs(dir_name: str) -> dict:
    directory = Path(dir_name)

    return {
        "relaxation_output": parse_relaxation_output(directory / "relaxation"),
        "optics_output": parse_optics_output(directory / "optics"),
        "linear_response": parse_out_lin(directory),
        "nonlinear_response": parse_out_nonlin(directory),
        "atomic_contributions": parse_atomic_contributions(directory / "out_nonlin"),
    }
