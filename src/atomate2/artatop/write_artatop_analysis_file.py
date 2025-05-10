from pathlib import Path

def write_artatop_analysis_file(output_model, cif_filename: str, filename_dir: str = "./"):
    """
    Write ART-{cif_name}.dat from SHG summary stored in output_model.shg_summary.
    """

    cif_stem = Path(cif_filename).stem
    filename = Path(filename_dir) / f"ART-{cif_stem}.dat"

    # Get SHG summary entries
    summary_entries = output_model.shg_summary or []

    # Get top tensor component and value (assume d33 is largest if not set)
    top_label = output_model.art_top_component or "d33"
    top_value = output_model.art_top_value or 0.0

    # Write file
    with open(filename, "w") as f:
        f.write(f"{cif_stem} {top_label} {top_value:.3f}\n")
        f.write(f"ART analysis results for the largest component {top_label} for optic_EgHSE\n")
        f.write("Type    Natom   IND     TOT     VB      CB      VB_s    VB_p    VB_d    CB_s    CB_p    CB_d    TOT_s   TOT_p   TOT_d   Atao\n")

        for e in summary_entries:
            atao = (e.vb_s + e.vb_p + e.vb_d + e.cb_s + e.cb_p + e.cb_d) / e.num_atoms
            f.write(
                f"{e.atom_type:<8}{e.num_atoms:<8}"
                f"{e.ind:<8.2f}{e.total:<8.2f}{e.vb / e.num_atoms:<8.2f}{e.cb / e.num_atoms:<8.2f}"
                f"{e.vb_s / e.num_atoms:<8.2f}{e.vb_p / e.num_atoms:<8.2f}{e.vb_d / e.num_atoms:<8.2f}"
                f"{e.cb_s / e.num_atoms:<8.2f}{e.cb_p / e.num_atoms:<8.2f}{e.cb_d / e.num_atoms:<8.2f}"
                f"{e.tot_s / e.num_atoms:<8.2f}{e.tot_p / e.num_atoms:<8.2f}{e.tot_d / e.num_atoms:<8.2f}"
                f"{atao:<8.2f}\n"
            )
