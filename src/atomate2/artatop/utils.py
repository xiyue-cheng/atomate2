import subprocess
from atomate2.artatop.artatop_parser import parse_artatop_outputs  
from pathlib import Path
import shutil
def run_artatop_shell_scripts(calc_dir: Path):
    
    script_base = Path(__file__).parent / "scripts"
    scripts = [
        "read_artatop-UV_lin",
        "read_artatop-UV_deff",
        "read_artatop-IR_lin",           
        "read_artatop-IR_deff",
    ]

    for script in scripts:
        src = script_base / script
        dest = calc_dir / script
        shutil.copy(src, dest)
        subprocess.run(["bash", str(dest)], cwd=calc_dir, check=True)
