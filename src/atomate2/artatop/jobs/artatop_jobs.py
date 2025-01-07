from pathlib import Path

from jobflow import Maker, job

from atomate2.artatop.run import run_artatop
from atomate2.artatop.sets.core import InputFileHandler


class LINMaker(Maker):
    """
    Maker for ARTATOP Linear (LIN) calculations.
    """

    name: str = "ARTATOP LIN Maker"
    output_dir: Path = Path("./artatop_outputs/out_lin")

    def __init__(self):
        """
        Initialize the LINMaker class and set up the InputFileHandler.
        """
        self.input_handler = InputFileHandler(output_dir=str(self.output_dir))

    @job
    def make(self, optics_dir):
        """
        Run LIN (linear) calculations.

        Parameters
        ----------
        optics_dir : Path
            Directory containing optics output.

        Returns
        -------
        dict
            Output path of the LIN calculation.
        """
        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate the LIN input file
        lin_input = self.input_handler.get_input_set("lin", self.output_dir)

        # Define the LIN output file
        lin_output = self.output_dir / "re_lin"

        # Directly run ARTATOP without Custodian
        command = f"artatop < {lin_input} > {lin_output}"
        run_artatop(job_type="direct", artatop_cmd=command, output_dir=self.output_dir)

        return {"lin_output": str(lin_output)}


class NLINMaker(Maker):
    """
    Maker for ARTATOP Non-Linear (NLIN) calculations.
    """

    name: str = "ARTATOP NLIN Maker"
    output_dir: Path = Path("./artatop_outputs/out_nlin")

    def __init__(self):
        """
        Initialize the NLINMaker class and set up the InputFileHandler.
        """
        self.input_handler = InputFileHandler(output_dir=str(self.output_dir))

    @job
    def make(self, optics_dir):
        """
        Run NLIN (non-linear) calculations.

        Parameters
        ----------
        optics_dir : Path
            Directory containing optics output.

        Returns
        -------
        dict
            Output path of the NLIN calculation.
        """
        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate the NLIN input file
        nlin_input = self.input_handler.get_input_set("nlin", self.output_dir)

        # Define the NLIN output file
        nlin_output = self.output_dir / "re_nlin"

        # Directly run ARTATOP without Custodian
        command = f"artatop < {nlin_input} > {nlin_output}"
        run_artatop(job_type="direct", artatop_cmd=command, output_dir=self.output_dir)

        return {"nlin_output": str(nlin_output)}


class ARTMaker(Maker):
    """
    Maker for ARTATOP ART calculations.
    """

    name: str = "ARTATOP ART Maker"
    output_dir: Path = Path("./artatop_outputs/out_nlin")  # Same as NLIN directory

    def __init__(self):
        """
        Initialize the ARTMaker class and set up the InputFileHandler.
        """
        self.input_handler = InputFileHandler(output_dir=str(self.output_dir))

    @job
    def make(self, nlin_output):
        """
        Run ART calculations.

        Parameters
        ----------
        nlin_output : str
            Path to the NLIN output file.

        Returns
        -------
        dict
            Output path of the ART calculation.
        """
        # Determine the highest contributing component
        component = self.input_handler.determine_highest_component(Path(nlin_output))

        # Generate the ART input file
        art_input = self.input_handler.get_input_set("art", self.output_dir, component)

        # Define the ART output file
        art_output = self.output_dir / "re_art"

        # Directly run ARTATOP without Custodian
        command = f"artatop < {art_input} > {art_output}"
        run_artatop(job_type="direct", artatop_cmd=command, output_dir=self.output_dir)

        return {"art_output": str(art_output)}
