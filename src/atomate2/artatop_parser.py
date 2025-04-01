import numpy as np
from pathlib import Path
from typing import Union, List
from atomate2.artatop.schemas import (
    LinearOpticalResponse,
    NonlinearOpticalResponse,
    DTensorValues,
    DeffValues,
    BirefringenceValues,
    AtomicContributions,
    BandEnergyInfo,
    DshgBandContribution,
    ArtatopOutputModel,
)
PI = np.pi

def parse_linear_response_at_energies(lin_dir: Union[str, Path], target_energies: List[float]) -> dict:
    def _parse_spin_set(spin_suffix=""):
        components = ["xx", "yy", "zz"]
        data_by_axis = {}

        for comp in components:
            path = lin_dir / f"lin_{comp}{spin_suffix}.dat"
            if not path.exists():
                continue

            in_dielectric_block = False
            dielectric_data = []

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#"):
                        in_dielectric_block = "Im(eps)" in line and "Re(eps)" in line
                        continue
                    if in_dielectric_block:
                        parts = line.split()
                        if len(parts) < 4:
                            continue
                        try:
                            energy = float(parts[0])
                            im_eps = float(parts[1])
                            re_eps = float(parts[2])
                            abs_eps = float(parts[3])
                            dielectric_data.append((energy, re_eps, im_eps, abs_eps))
                        except ValueError:
                            continue

            if dielectric_data:
                dielectric_data = np.array(dielectric_data)
                data_by_axis[comp] = {
                    "energy": dielectric_data[:, 0],
                    "real": dielectric_data[:, 1],
                    "imag": dielectric_data[:, 2],
                    "abs": dielectric_data[:, 3],
                }

        responses, birefringence_list = [], []

        for target_energy in target_energies:
            n_vals = {}
            for comp, data in data_by_axis.items():
                if len(data["energy"]) == 0:
                    continue
                idx = np.abs(data["energy"] - target_energy).argmin()
                eps_real = data["real"][idx]
                eps_imag = data["imag"][idx]
                n = np.sqrt((np.sqrt(eps_real ** 2 + eps_imag ** 2) + eps_real) / 2)
                n_vals[comp] = n
                responses.append(LinearOpticalResponse(
                    energy=data["energy"][idx],
                    real_part=eps_real,
                    imaginary_part=eps_imag,
                    absorption_coefficient=float(data["abs"][idx]),
                    refractive_index=n,
                    extinction_coefficient=None
                ))

            if n_vals:
                delta_n = max(n_vals.values()) - min(n_vals.values())
                birefringence_list.append(BirefringenceValues(energy=target_energy, delta_n=delta_n))

        return {
            "linear": responses,
            "birefringence": birefringence_list
        }

    lin_dir = Path(lin_dir)
    return {
        "base": _parse_spin_set(""),
        "up": _parse_spin_set("_up"),
        "down": _parse_spin_set("_down")
    }

def parse_nonlinear_response_at_energies(nonlin_dir: Union[str, Path], target_energies: List[float]) -> dict:
    def _parse_spin_set(spin_suffix=""):
        files = sorted(nonlin_dir.glob(f"nonlin_*{spin_suffix}.dat"))
        if not files:
            return None

        component_data = {}
        for file in files:
            label = file.name.replace(f"nonlin_", "").replace(f"{spin_suffix}.dat", "")
            energies, im_vals, re_vals = [], [], []
            with open(file) as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) < 3:
                            continue
                        try:
                            energies.append(float(parts[0]))
                            im_vals.append(float(parts[1]))
                            re_vals.append(float(parts[2]))
                        except ValueError:
                            continue
            component_data[label] = {
                "energy": np.array(energies),
                "im": np.array(im_vals),
                "re": np.array(re_vals)
            }

        if not component_data:
            return None

        nonlinear_responses, deff_results, d_tensor_results = [], [], []

        for target_energy in target_energies:
            total_re, total_im = 0.0, 0.0
            d_tensor = {}
            for comp, data in component_data.items():
                idx = np.abs(data["energy"] - target_energy).argmin()
                chi2_re = data["re"][idx]
                chi2_im = data["im"][idx]
                d_val = (chi2_re / 2.0) * (4 * PI / 3.0) * 10
                d_tensor[f"d_{comp}"] = d_val
                total_re += chi2_re
                total_im += chi2_im

            deff = np.sqrt(sum(v ** 2 for v in d_tensor.values()))

            nonlinear_responses.append(NonlinearOpticalResponse(
                energy=target_energy,
                total_real_part=total_re,
                total_imaginary_part=total_im
            ))
            deff_results.append(DeffValues(energy=target_energy, deff=deff))
            d_tensor_results.append(DTensorValues(energy=target_energy, components=d_tensor))

        return {
            "nonlinear": nonlinear_responses,
            "deff": deff_results,
            "d_tensor": d_tensor_results
        }

    nonlin_dir = Path(nonlin_dir)
    base = _parse_spin_set("")
    up = _parse_spin_set("_up")
    down = _parse_spin_set("_down")

    return {
        "base": base or {
            "nonlinear": [],
            "deff": [],
            "d_tensor": []
        },
        "up": up or {
            "nonlinear": [],
            "deff": [],
            "d_tensor": []
        },
        "down": down or {
            "nonlinear": [],
            "deff": [],
            "d_tensor": []
        }
    }
      
def parse_lin_nlin_outputs(dir_name: str, energies: list[float] = [0.0, 0.65, 1.167]) -> ArtatopOutputModel:
    dir_path = Path(dir_name)
    lin = parse_linear_response_at_energies(dir_path / "out_lin", energies)
    nlin = parse_nonlinear_response_at_energies(dir_path / "out_nonlin", energies)

    return ArtatopOutputModel(
        dir_name=str(dir_path),

        # Base (non-spin)
        linear_response=lin["base"]["linear"],
        nonlinear_response=nlin["base"]["nonlinear"],
        d_tensor=nlin["base"]["d_tensor"],
        deff_values=nlin["base"]["deff"],
        birefringence=lin["base"]["birefringence"],

        # Spin-up (optional)
        linear_response_up=lin["up"]["linear"] if lin["up"] else None,
        d_tensor_up=nlin["up"]["d_tensor"] if nlin["up"] else None,
        deff_values_up=nlin["up"]["deff"] if nlin["up"] else None,
        birefringence_up=lin["up"]["birefringence"] if lin["up"] else None,

        # Spin-down (optional)
        linear_response_down=lin["down"]["linear"] if lin["down"] else None,
        d_tensor_down=nlin["down"]["d_tensor"] if nlin["down"] else None,
        deff_values_down=nlin["down"]["deff"] if nlin["down"] else None,
        birefringence_down=lin["down"]["birefringence"] if lin["down"] else None
    )
    
    
def write_result_re(output_model: ArtatopOutputModel, filename="result.re"):
    def write_block(f, label, d_tensor, deff_values, lin_resp, biref):
        f.write(f"\n{label}\n") 
        for deff in deff_values:
            f.write(f"deff at omega = {deff.energy} eV\n")
            f.write(f"{deff.deff:.3f}\n\n")

        for dset in d_tensor:
            f.write(f"d at omega = {dset.energy} eV\n")
            d = dset.components
            rows = [
                [d.get("d_xxx", 0), d.get("d_xyy", 0), d.get("d_xzz", 0), d.get("d_xyz", 0), d.get("d_xxz", 0), d.get("d_xxy", 0)],
                [d.get("d_yxx", 0), d.get("d_yyy", 0), d.get("d_yzz", 0), d.get("d_yyz", 0), d.get("d_yxz", 0), d.get("d_yxy", 0)],
                [d.get("d_zxx", 0), d.get("d_zyy", 0), d.get("d_zzz", 0), d.get("d_zyz", 0), d.get("d_zxz", 0), d.get("d_zxy", 0)],
            ]
            for row in rows:
                f.write(" ".join(f"{x:.3f}" for x in row) + "\n")
            f.write("\n")

        for energy in [x.energy for x in deff_values]:
            matching_resp = [r for r in lin_resp if abs(r.energy - energy) < 1e-3]
            e = [r.real_part for r in matching_resp]
            n = [r.refractive_index for r in matching_resp]
            biref_val = next((b.delta_n for b in biref if abs(b.energy - energy) < 1e-3), None)

            f.write(f"linear optic epsilon at {energy:.3f} eV\n")
            f.write(" ".join(f"{x:.3f}" for x in e) + "\n\n")
            f.write(f"refractive n at {energy:.3f} eV\n")
            f.write(" ".join(f"{x:.3f}" for x in n) + "\n\n")
            f.write(f"birefringence at {energy:.3f} eV\n")
            f.write(f"{biref_val:.3f}\n\n")

    with open(filename, "w") as f:
        # base
        write_block(
            f,
            label="Base (no spin)",
            d_tensor=output_model.d_tensor,
            deff_values=output_model.deff_values,
            lin_resp=output_model.linear_response,
            biref=output_model.birefringence,
        )

        # Spin-up
        if output_model.d_tensor_up and output_model.deff_values_up:
            write_block(
                f,
                label="Spin Up",
                d_tensor=output_model.d_tensor_up,
                deff_values=output_model.deff_values_up,
                lin_resp=output_model.linear_response_up,
                biref=output_model.birefringence_up,
            )

        # Spin-down
        if output_model.d_tensor_down and output_model.deff_values_down:
            write_block(
                f,
                label="Spin Down",
                d_tensor=output_model.d_tensor_down,
                deff_values=output_model.deff_values_down,
                lin_resp=output_model.linear_response_down,
                biref=output_model.birefringence_down,
            )  

from pathlib import Path
from pymatgen.core import Structure
from atomate2.artatop.schemas import AtomicContributions, OrbitalContribution


def parse_orbital_atomic_contributions(structure: Structure, out_dir: Path) -> List[AtomicContributions]:
    """
    Parse and sum orbital contributions from all SHG val/con files per atom.
    """
    num_atoms = len(structure)
    atom_types = structure.species
    val_files = sorted(out_dir.glob("arp_shg_val_*.txt"))
    con_files = sorted(out_dir.glob("arp_shg_con_*.txt"))

    if not val_files or not con_files:
        raise FileNotFoundError("No SHG val/con files found in out_nonlin.")

    def read_shg_file(filepath):
        lines = filepath.read_text().splitlines()
        data_lines = [line for line in lines if line.strip() and not line.startswith("#")]
        return [[float(x) for x in line.split()] for line in data_lines]

    summed_val = [[0.0, 0.0, 0.0] for _ in range(num_atoms)]
    summed_con = [[0.0, 0.0, 0.0] for _ in range(num_atoms)]

    for vf in val_files:
        val_data = read_shg_file(vf)
        for i in range(num_atoms):
            for j in range(3):
                summed_val[i][j] += val_data[i][j]

    for cf in con_files:
        con_data = read_shg_file(cf)
        for i in range(num_atoms):
            for j in range(3):
                summed_con[i][j] += con_data[i][j]

    atomic_contribs = []

    for i in range(num_atoms):
        atom_index = i + 1
        species = str(atom_types[i])

        orbital_dict = {
            "s": summed_val[i][0] + summed_con[i][0],
            "p": summed_val[i][1] + summed_con[i][1],
            "d": summed_val[i][2] + summed_con[i][2],
        }

        atomic_contribs.append(AtomicContributions(
            atom=f"{atom_index}-{species}",
            orbital_contributions=orbital_dict,
            total_contribution=sum(orbital_dict.values())
        ))

    return atomic_contribs
    
def write_result_art_IND(contributions: list[AtomicContributions], filename: Path = Path("result.art_IND")):
    with open(filename, "w") as f:
        header = f"{'Atom':<10} {'Total':>10} | {'s':>8} {'p':>8} {'d':>8}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for contrib in contributions:
            orb_map = contrib.orbital_contributions
            f.write(f"{contrib.atom:<10} {contrib.total_contribution:10.3f} |"
                    f"{orb_map.get('s', 0):8.3f}{orb_map.get('p', 0):8.3f}{orb_map.get('d', 0):8.3f}\n")  
                    
                    
def parse_band_structure_energy_info(
    procar_path: Path,
    outcar_path: Path,
    output_file: Path = Path("energy-band.dat")
) -> list[BandEnergyInfo]:
    """
    Parse band energies from PROCAR, align them with the Fermi energy,
    and write a flat summary file like read_procar.
    """
    from monty.io import zopen
    from atomate2.artatop.schemas import BandEnergyInfo

    # Step 1: Extract E-fermi
    with open(outcar_path) as f:
        for line in f:
            if "E-fermi" in line:
                efermi = float(line.split()[2])
                break
        else:
            raise ValueError("Fermi energy not found in OUTCAR")

    with zopen(procar_path, "rt") as f:
        lines = f.readlines()

    energies_by_band = {}
    current_band = None

    for line in lines:
        if line.strip().startswith("band"):
            parts = line.strip().split()
            band_number = int(parts[1])
            energy = float(parts[4])
            energies_by_band.setdefault(band_number, []).append(energy)

    band_infos = []

    with open(output_file, "w") as fout:
        for band_number, energies in energies_by_band.items():
            shifted = [round(e - efermi, 3) for e in energies]
            e_max = max(shifted)
            e_min = min(shifted)
            last_energy = shifted[-1]
            e_prf = e_max if last_energy < 0 else e_min

            fout.write(f"{band_number} {e_prf:.3f} {e_max:.3f} {e_min:.3f}\n")
            band_infos.append(BandEnergyInfo(
                band_index=band_number,
                e_prf=e_prf,
                e_max=e_max,
                e_min=e_min
            ))

    return band_infos 

def process_and_write_dshg_data(
    total_file: Path,
    vb_file: Path,
    cb_file: Path,
    out_dir: Path
) -> list[DshgBandContribution]:
    """
    Process SHG contributions, write both full and minimal dshg-PmV output files,
    and return structured data for JSON.
    """

    def read_data(file_path: Path):
        data = []
        with open(file_path) as f:
            for line in f:
                if line.strip().startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    data.append((float(parts[1]), float(parts[2])))
        return data

    total_data = read_data(total_file)
    vb_data = read_data(vb_file)
    cb_data = read_data(cb_file)

    summary = []
    full_file = out_dir / "dshg-PmV-full.dat"
    minimal_file = out_dir / "dshg-PmV.dat"

    with open(full_file, "w") as f_full, open(minimal_file, "w") as f_min:
        f_full.write(f"# {'Band':<5} {'Im(χ_total)':>12} {'Re(χ_total)':>12}  "
                     f"{'Im(χ_vb)':>12} {'Re(χ_vb)':>12}  "
                     f"{'Im(χ_cb)':>12} {'Re(χ_cb)':>12}\n")

        for i, total in enumerate(total_data):
            im_tot, re_tot = total
            im_vb, re_vb = vb_data[i] if i < len(vb_data) else (0.0, 0.0)
            im_cb, re_cb = cb_data[i] if i < len(cb_data) else (0.0, 0.0)

            # Full version
            f_full.write(f"{i+1:<6} {im_tot:12.6E} {re_tot:12.6E}  "
                         f"{im_vb:12.6E} {re_vb:12.6E}  "
                         f"{im_cb:12.6E} {re_cb:12.6E}\n")

            # Minimal version (Im(χ_total) only)
            f_min.write(f"{i+1} {im_tot:.6f}\n")

            summary.append(DshgBandContribution(
                band_index=i + 1,
                im_total=im_tot,
                re_total=re_tot,
                im_vb=im_vb,
                re_vb=re_vb,
                im_cb=im_cb,
                re_cb=re_cb
            ))

    return summary
def parse_artatop_outputs(dir_name: str, energies: list[float] = [0.0, 0.65, 1.167]) -> ArtatopOutputModel:
    dir_path = Path(dir_name)

    # Step 1: parse linear + nonlinear outputs
    lin = parse_linear_response_at_energies(dir_path / "out_lin", energies)
    nlin = parse_nonlinear_response_at_energies(dir_path / "out_nonlin", energies)

    # Step 2: atomic orbital contributions
    structure = Structure.from_file(dir_path / "POSCAR")
    out_dir = dir_path / "out_nonlin"
    atomic_contribs = parse_orbital_atomic_contributions(structure, out_dir)

    # Step 3: generate result.art_IND
    write_result_art_IND(atomic_contribs, filename=dir_path / "result.art_IND")
    
    band_energy_info = parse_band_structure_energy_info(
        procar_path=Path(dir_name) / "PROCAR",
        outcar_path=Path(dir_name) / "OUTCAR",
        output_file=Path(dir_name) / "energy-band.dat",
     
    )
    
    dshg_json = process_and_write_dshg_data(
        total_file=dir_path / "out_nonlin" / "arp_dshg_xyz.txt",
        vb_file=dir_path / "out_nonlin" / "arp_dshg-vb_xyz.txt",
        cb_file=dir_path / "out_nonlin" / "arp_dshg-cb_xyz.txt",
        out_dir=dir_path / "out_nonlin"
    )


    # Step 4: return full output model
    return ArtatopOutputModel(
        dir_name=str(dir_path),
        linear_response=lin["base"]["linear"],
        nonlinear_response=nlin["base"]["nonlinear"],
        d_tensor=nlin["base"]["d_tensor"],
        deff_values=nlin["base"]["deff"],
        birefringence=lin["base"]["birefringence"],

        linear_response_up=lin["up"]["linear"] if lin["up"] else None,
        d_tensor_up=nlin["up"]["d_tensor"] if nlin["up"] else None,
        deff_values_up=nlin["up"]["deff"] if nlin["up"] else None,
        birefringence_up=lin["up"]["birefringence"] if lin["up"] else None,

        linear_response_down=lin["down"]["linear"] if lin["down"] else None,
        d_tensor_down=nlin["down"]["d_tensor"] if nlin["down"] else None,
        deff_values_down=nlin["down"]["deff"] if nlin["down"] else None,
        birefringence_down=lin["down"]["birefringence"] if lin["down"] else None,

        atomic_contributions=atomic_contribs
    )
