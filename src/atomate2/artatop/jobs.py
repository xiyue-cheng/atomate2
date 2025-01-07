from pathlib import Path

from jobflow import Maker, job

from atomate2.artatop.files import generate_artatop_inputs
from atomate2.artatop.parser import parse_artatop_outputs
from atomate2.artatop.run1 import run_artatop_direct
from atomate2.artatop.schemas import ArtatopTaskDocument


class ARTATOPMaker(Maker):
    """
    Maker for running ARTATOP calculations directly (without Custodian).
    """

    name = "ARTATOP Maker"

    def __init__(
        self,
        calc_type: str,
        output_dir: str | Path = "./artatop_outputs",
        parameters: dict[str, float] | None = None,
        src_dir: str | None = None,
        custom_components: str | None = None,
    ):
        self.calc_type = calc_type
        self.output_dir = Path(output_dir).resolve()
        self.parameters = parameters or {}
        self.src_dir = src_dir
        self.custom_components = custom_components

    @job(output_schema=ArtatopTaskDocument)
    def make(self) -> ArtatopTaskDocument:
        """
        Run an ARTATOP calculation directly and return the task document.

        Returns
        -------
        ArtatopTaskDocument
            Parsed results from ARTATOP calculations.
        """
        # Generate the input file
        input_file = generate_artatop_inputs(
            calculation_type=self.calc_type,
            parameters=self.parameters,
            src_dir=self.src_dir,
            custom_components=self.custom_components,
        )

        # Construct ARTATOP command
        artatop_cmd = (
            f"artatop < {input_file} > {self.output_dir}/output_{self.calc_type}"
        )

        # Run ARTATOP directly
        run_artatop_direct(
            artatop_cmd=artatop_cmd,
            output_dir=self.output_dir,
            input_file=input_file,
        )

        # Parse the outputs using the parser function
        parsed_outputs = parse_artatop_outputs(self.output_dir)

        # Return the parsed outputs as a task document
        return ArtatopTaskDocument.from_parsed_outputs(parsed_outputs)
