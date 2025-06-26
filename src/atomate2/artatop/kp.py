import pathlib
from typing import Optional
import gzip
import xml.etree.ElementTree as ET
def load_kpoints_from_vasprun(vasprun_path: pathlib.Path) -> Optional[list[int]]:
    """
    Extract the automatic k-point grid (divisions) from a vasprun.xml or .gz file.

    Args:
        vasprun_path (pathlib.Path): Path to vasprun.xml or vasprun.xml.gz

    Returns:
        Optional[list[int]]: K-point grid like [12, 12, 12], or None if not found.
    """
    if not vasprun_path.exists():
        print(f"WARNING: vasprun file not found: {vasprun_path}")
        return None

    try:
        # Load XML content
        if vasprun_path.suffix == ".gz":
            with gzip.open(vasprun_path, 'rt') as f:
                tree = ET.parse(f)
        else:
            tree = ET.parse(vasprun_path)

        root = tree.getroot()
        for gen in root.iter("generation"):
            if gen.attrib.get("param") in ("Gamma", "kpts"):
                for v in gen.findall("v"):
                    if v.attrib.get("name") == "divisions":
                        divisions = [int(x) for x in v.text.strip().split()]
                        if len(divisions) >= 3:
                            return divisions[:3]
        print(f"WARNING: No <generation> divisions found.")
        return None

    except Exception as e:
        print(f"WARNING: Manual parsing of vasprun.xml failed: {e}")
        return None
