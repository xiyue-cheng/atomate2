from pathlib import Path


class InputFileHandler:
    """
    A handler for generating ARTATOP input files: input_lin, input_nlin, and input_art.
    """

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
            (calc_dir/ "out_lin").mkdir(exist_ok=True)
        elif calc_type == "nlin":
            content = """NO 777
$dft_src lvasp=T $end
$opc maxomega = 30  domega = 0.01167  scissor=0.00  ecutmin = 0.03  smear = 0.03 $end
"""
            (calc_dir/ "out_nonlin").mkdir(exist_ok=True)
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
        

    def find_best_component_from_nonlin(self, nonlin_dir: Path) -> str:
        """
        From all nonlin_*.dat files, find the component with the highest
        Tot-Re Chi(-2w,w,w) value (first data row, 3rd column).
        """
        axis_map = {"x": "1", "y": "2", "z": "4"}

        def component_from_filename(filename: str) -> str:
            base = filename.replace("nonlin_", "").replace(".dat", "")
            return ''.join(axis_map[c] for c in base)

        def read_first_value(file: Path) -> float:
            with open(file) as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        parts = line.split()
                        return float(parts[2])  # 3rd column
            raise ValueError(f"No valid data in {file.name}")

        best_value = float("-inf")
        best_component = None

        for file in nonlin_dir.glob("nonlin_*.dat"):
            try:
                value = read_first_value(file)
                component = component_from_filename(file.name)
                if value > best_value:
                    best_value = value
                    best_component = component
            except Exception as e:
                print(f"Skipping {file.name}: {e}")

        if best_component is None:
            raise RuntimeError("Could not determine best component from nonlin files.")

        return best_component

    def get_best_component(self, calc_dir: Path) -> str:
        """
        Get the best component from result.re if it exists, otherwise use the default method.
        """
        result_re = calc_dir / "result.re"
        if result_re.exists():
            try:
                print("Using result.re to determine best component...")
                return self.determine_highest_component(result_re)
            except Exception as e:
                print(f"Failed to use result.re: {e}")
                print("Falling back to nonlin_*.dat files.")
        
        print("Using nonlin_*.dat files to determine best component...")
        return self.find_best_component_from_nonlin(calc_dir / "out_nonlin")
