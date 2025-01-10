"""
Module defines job makers for ARTATOP workflows.

It includes:
- LINMaker: For linear optical response calculations.
- NLINMaker: For nonlinear optical response calculations.
- ARTMaker: For atomic response calculations.
"""

from pathlib import Path

from jobflow import Maker, job

from atomate2.artatop.run import run_artatop
from atomate2.artatop.sets.core import InputFileHandler


class LINMaker(Maker):
    """Maker for ARTATOP Linear (LIN) calculations."""

    name: str = "ARTATOP LIN Maker"

    def __init__(self) -> None:
        """Initialize the LINMaker class."""
        self.input_handler: InputFileHandler  # Properly annotate the type

    @job
    def make(self, optics_dir: str, base_dir: str) -> dict:
        """
        Run LIN (linear) calculations.

        Parameters
        ----------
        optics_dir : str
            Directory containing optics output.
        base_dir : str
            Base directory for storing LIN outputs.

        Returns
        -------
        dict
            Output path of the LIN calculation.
        """
        output_dir = Path(base_dir) / "out_lin"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.input_handler = InputFileHandler(output_dir=str(output_dir))
        lin_input = self.input_handler.get_input_set("lin", output_dir)
        lin_output = output_dir / "re_lin"

        command = f"artatop < {lin_input} > {lin_output}"
        run_artatop(job_type="direct", artatop_cmd=command)

        return {"lin_output": str(lin_output)}


class NLINMaker(Maker):
    """Maker for ARTATOP Non-Linear (NLIN) calculations."""

    name: str = "ARTATOP NLIN Maker"

    def __init__(self) -> None:
        """Initialize the NLINMaker class."""
        self.input_handler: InputFileHandler  # Properly annotate the type

    @job
    def make(self, lin_output: str, base_dir: str) -> dict:
        """
        Run NLIN (non-linear) calculations.

        Parameters
        ----------
        lin_output : str
            Path to the LIN output file.
        base_dir : str
            Base directory for storing NLIN outputs.

        Returns
        -------
        dict
            Output path of the NLIN calculation.
        """
        output_dir = Path(base_dir) / "out_nonlin"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.input_handler = InputFileHandler(output_dir=str(output_dir))
        nlin_input = self.input_handler.get_input_set("nlin", output_dir)
        nlin_output = output_dir / "re_nlin"

        command = f"artatop < {nlin_input} > {nlin_output}"
        run_artatop(job_type="direct", artatop_cmd=command)

        return {"nlin_output": str(nlin_output)}


class ARTMaker(Maker):
    """Maker for ARTATOP ART calculations."""

    name: str = "ARTATOP ART Maker"

    def __init__(self) -> None:
        """Initialize the ARTMaker class."""
        self.input_handler: InputFileHandler  # Properly annotate the type

    @job
    def make(self, nlin_output: str, base_dir: str) -> dict:
        """
        Run ART calculations.

        Parameters
        ----------
        nlin_output : str
            Path to the NLIN output file.
        base_dir : str
            Base directory for storing ART outputs.

        Returns
        -------
        dict
            Output path of the ART calculation.
        """
        output_dir = Path(base_dir) / "out_nonlin"  # Same as NLIN outputs
        output_dir.mkdir(parents=True, exist_ok=True)

        self.input_handler = InputFileHandler(output_dir=str(output_dir))
        component = self.input_handler.determine_highest_component(Path(nlin_output))
        art_input = self.input_handler.get_input_set("art", output_dir, component)
        art_output = output_dir / "re_art"

        command = f"artatop < {art_input} > {art_output}"
        run_artatop(job_type="direct", artatop_cmd=command)

        return {"art_output": str(art_output)}
