from pathlib import Path


class InputFileHandler:
    """
    A handler for generating ARTATOP input files: input_lin, input_nlin, and input_art.
    """

    def __init__(self, output_dir: str = "./artatop_outputs"):
        """
        Initialize the InputFileHandler.

        Parameters
        ----------
        output_dir : str
            Directory to store input files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_input_set(
        self, calc_type: str, calc_dir: Path, component: str = None
    ) -> Path:
        """
        Generate the appropriate input file for the given calculation type.

        Parameters
        ----------
        calc_type : str
            Type of calculation (e.g., "lin", "nlin", "art").
        calc_dir : Path
            Directory where the input file will be created.
        component : str, optional
            Component for the ART calculation, by default None.

        Returns
        -------
        Path
            Path to the generated input file.
        """
        calc_dir.mkdir(parents=True, exist_ok=True)

        if calc_type == "lin":
            content = """LO 77
$dft_src lvasp=T $end
$opc maxomega = 30  domega = 0.01167  scissor=0.00  ecutmin = 0.03  smear = 0.03 $end
"""
        elif calc_type == "nlin":
            content = """NO 777
$dft_src lvasp=T $end
$opc maxomega = 30  domega = 0.01167  scissor=0.00  ecutmin = 0.03  smear = 0.03 $end
"""
        elif calc_type == "art":
            if not component:
                raise ValueError("Component is required for ART calculations.")
            content = f"""AR {component}
$dft_src lvasp=T $end
$opc maxomega = 30  domega = 0.01167  scissor=0.00  ecutmin = 0.03  smear = 0.03 $end
"""
        else:
            raise ValueError(f"Unsupported calculation type: {calc_type}")

        input_file = calc_dir / f"input_{calc_type}"
        with open(input_file, "w") as f:
            f.write(content)

        return input_file

    def determine_highest_component(self, result_re_file: Path) -> str:
        """
        Determine the highest component for ARTATOP calculation from the first T-matrix block.

        Parameters
        ----------
        result_re_file : Path
            Path to the `result.re` file.

        Returns
        -------
        str
            Component with the highest value (e.g., "311").
        """
        if not result_re_file.exists():
            raise FileNotFoundError(
                f"`{result_re_file}` not found. Please provide a valid `result.re` file."
            )

        # Define matrix-to-component mapping for a 3x6 matrix
        matrix_to_component = {
            (1, 1): "111",
            (1, 2): "122",
            (1, 3): "133",
            (1, 4): "123",
            (1, 5): "113",
            (1, 6): "112",
            (2, 1): "211",
            (2, 2): "222",
            (2, 3): "233",
            (2, 4): "223",
            (2, 5): "213",
            (2, 6): "212",
            (3, 1): "311",
            (3, 2): "322",
            (3, 3): "333",
            (3, 4): "323",
            (3, 5): "313",
            (3, 6): "312",
        }

        max_value = float("-inf")
        max_position = None

        with open(result_re_file) as f:
            lines = f.readlines()

        # Locate the first T-matrix block starting with "d at omege = 0 eV"
        matrix_start = None
        for idx, line in enumerate(lines):
            if line.startswith("d at omege = 0 eV"):
                matrix_start = (
                    idx + 1
                )  # The next line should be the start of the matrix
                break

        if matrix_start is None:
            raise ValueError(
                "T-matrix for `d at omege = 0 eV` not found in `result.re`."
            )

        # Parse the 3x6 matrix following the header
        for row_idx in range(3):  # Expecting 3 rows
            values = list(map(float, lines[matrix_start + row_idx].split()))
            for col_idx, value in enumerate(values):
                if value > max_value:
                    max_value = value
                    max_position = (row_idx + 1, col_idx + 1)

        if max_position in matrix_to_component:
            return matrix_to_component[max_position]
        raise ValueError("Invalid T-matrix format or unexpected data in `result.re`.")
