"""Module defining flow-related utilities for ARTATOP workflows."""

from jobflow import run_locally
from jobflow_remote import submit_flow
from pymatgen.core import Structure

from atomate2.artatop.flows.workflow import ArtatopWorkflowMaker

# Define the structure (update this with your actual structure)
structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)

# Define the workflow maker
workflow_maker = ArtatopWorkflowMaker()

# Create the workflow
artatop_flow = workflow_maker.make(structure=structure)
# run the job
run_locally(artatop_flow, create_folders=True)

# resIurces = {"nodes": 1, "ntasks": 4, "partition": "batch","time":"1:00:00"}
resources = {"nodes": 1, "partition": "mars", "time": "1:00:00"}
# exec_config="yhrun -N 1 -p cp6"
submit_flow(artatop_flow, project="atlas", worker="tianhe_worker", resources=resources)
# resources = {"cores": 4 }
# exec_config="mpirun -n 4"
# submit_flow(lobster,project="atlas",worker="th-ex-ln1_worker",resources=resources)
