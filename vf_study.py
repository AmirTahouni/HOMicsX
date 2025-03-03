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
    init_data = load_json(JSON_INPUT_PATH)

    vf_list = [0.05, 0.1, 0.2, 0.25]
    outputs = []

    for vf in vf_list:
        error = -1
        
        while error < -0.02:# or error > 0.5:
            r = calculate_radius(vf, num_particles_)
            geometry = generate_random_geometry(vf, num_particles_, r, delDistanceMultiplier_, delDistanceBorder_)

            mesh, ct, ft = generate_mesh(geometry, r, min_size_, max_size_, view=False)

            # Solve the FEM problem and process the results
            C_hom = fem_solve(mesh, ct, ft, E_particle_, E_matrix_, nu_particle_, nu_matrix_)
            output = post_process(C_hom, E_particle_, E_matrix_, nu_particle_, nu_matrix_, vf)
            
            # Extract kappa values from the output
            kappa_res = output["Result"]["kappa_eff"]
            kappa_dil = output["WILLIS-Dilute"]["kappa_eff"]
            
            # Calculate the error between the results
            if kappa_dil != 0:  # Prevent division by zero
                error = (kappa_res - kappa_dil) / kappa_dil
            else:
                error = float('inf')  # If kappa_dil is zero, set error to infinity
            print(f"Volume fraction: {vf}, Error: {error:.4f}")

        outputs.append(output)
        print(f"Results for vf={vf} accepted with error: {error:.4f}")


    # Extract results from the 'outputs' list
    E_eff_list = []
    nu_eff_list = []
    kappa_eff_list = []
    mu_eff_list = []

    E_dilute_list = []
    nu_dilute_list = []
    kappa_dilute_list = []
    mu_dilute_list = []

    E_SC_list = []
    nu_SC_list = []
    kappa_SC_list = []
    mu_SC_list = []

    kappa_lower_list = []
    kappa_upper_list = []
    mu_lower_list = []
    mu_upper_list = []

    # Iterate over each dictionary in the 'outputs' list
    for result in outputs:
        E_eff_list.append(result["Result"]["E_eff"])
        nu_eff_list.append(result["Result"]["nu_eff"])
        kappa_eff_list.append(result["Result"]["kappa_eff"])
        mu_eff_list.append(result["Result"]["mu_eff"])

        E_dilute_list.append(result["WILLIS-Dilute"]["E_eff"])
        nu_dilute_list.append(result["WILLIS-Dilute"]["nu_eff"])
        kappa_dilute_list.append(result["WILLIS-Dilute"]["kappa_eff"])
        mu_dilute_list.append(result["WILLIS-Dilute"]["mu_eff"])

        E_SC_list.append(result["WILLIS-SelfConsistant"]["E_eff"])
        nu_SC_list.append(result["WILLIS-SelfConsistant"]["nu_eff"])
        kappa_SC_list.append(result["WILLIS-SelfConsistant"]["kappa_eff"])
        mu_SC_list.append(result["WILLIS-SelfConsistant"]["mu_eff"])

        kappa_lower_list.append(result["Hashin-Strikman"]["kappa_lower"])
        kappa_upper_list.append(result["Hashin-Strikman"]["kappa_upper"])
        mu_lower_list.append(result["Hashin-Strikman"]["mu_lower"])
        mu_upper_list.append(result["Hashin-Strikman"]["mu_upper"])
    
    E_norm = []
    for E in E_eff_list:
        E_norm.append(E/E_matrix_)

    plt.figure(figsize=(12,6))
    plt.scatter(vf_list, E_norm, color='b', marker='o')

    plt.xlabel("Volume Fraction")
    plt.ylabel("E_eff/E_matrix")
    plt.grid(True)
    plt.savefig("report/EE.pdf")
    plt.show(block=False)
    input("Press any key to exit...")


    # # Plot the results
    # plt.figure(figsize=(12, 6))
    #
    # # Plot Effective Modulus (E_eff) for Homogenized, Dilute, and SC Methods
    # plt.scatter(vf_list, kappa_eff_list, 
    #             label=r'$\kappa_{\mathrm{eff}}$ (Homogenized)', 
    #             color='b', marker='o')
    # plt.scatter(vf_list, kappa_dilute_list, 
    #             label=r'$\kappa_{\mathrm{eff}}$ (Willis Dilute)', 
    #             color='r', marker='x')
    # plt.scatter(vf_list, kappa_SC_list, 
    #             label=r'$\kappa_{\mathrm{eff}}$ (Willis Self-Consistent)', 
    #             color='g', marker='s')
    #
    # # Plot Hashin-Strikman Bounds for kappa (Bulk Modulus)
    # plt.fill_between(vf_list, kappa_lower_list, kappa_upper_list, color='gray', alpha=0.5, label="Hashin-Strikman Bulk Modulus Bounds")
    #
    # # Adding labels and title
    # plt.xlabel("Volume Fraction")
    # plt.ylabel("Effective Bulk Moduli (Pa)")
    # plt.title("Study of Effective Bulk Moduli vs Volume Fraction")
    # plt.legend(loc="best")
    # plt.grid(True)
    # plt.savefig('report/figure.pdf')
    # plt.show(block=False)
    # input("Press Enter to close the plot and end the run...")
    #
    #

