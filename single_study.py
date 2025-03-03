import numpy as np
import matplotlib.pyplot as plt

from config import *
from utils import *
from geometry import generate_random_geometry
from mesh import generate_mesh
from fem import fem_solve
from post import post_process

#-------------------------------------------------| input parsing |-------------------------------------#
init_data = load_json(JSON_INPUT_PATH)

elastic_props = init_data['input']['materialProperties']['linearElastic']
dist_props = init_data['input']['materialProperties']['particleDistribution']
geometrySetitngs = init_data['input']['geometryParameters']
mesh_props = init_data['input']['meshSettings']

E_particle_ = elastic_props['E_particle']
E_matrix_ = elastic_props['E_matrix']
nu_particle_ = elastic_props['nu_particle']
nu_matrix_ = elastic_props['nu_matrix']

volume_fraction_ = dist_props['volume_fraction']
num_particles_ = dist_props['num_particles']

delDistanceMultiplier_ =  geometrySetitngs['delDistanceMultiplier']
delDistanceBorder_ = geometrySetitngs['delDistanceBorder']

min_size_ = mesh_props['minSize']
max_size_ = mesh_props['maxSize']

r_ = calculate_radius(volume_fraction_, num_particles_)



#-------------------------------------------------| main |-------------------------------------------------#
if __name__ == "__main__":
    geometry = generate_random_geometry(volume_fraction_, num_particles_, r_, delDistanceMultiplier_, delDistanceBorder_)

    mesh, ct, ft = generate_mesh(geometry, r_, min_size_, max_size_, view=False)

    # Solve the FEM problem and process the results
    C_hom = fem_solve(mesh, ct, ft, E_particle_, E_matrix_, nu_particle_, nu_matrix_)
    output = post_process(C_hom, E_particle_, E_matrix_, nu_particle_, nu_matrix_, volume_fraction_)

    init_data["output"] = output
    save_json(path= JSON_INPUT_PATH, data= init_data)
