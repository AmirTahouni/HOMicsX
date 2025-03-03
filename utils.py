from config import *
import json
import numpy as np

#-------------------------------------------------| utils |------------------------------------------------#
def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)

def save_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def E_nu_to_kappa_mu(E, nu):
    """
    Converts Young's modulus (E) and Poisson's ratio (nu) to shear modulus (mu) and bulk modulus (kappa).

    Parameters:
        E (float or np.ndarray): Young's modulus.
        nu (float or np.ndarray): Poisson's ratio.
    
    Returns:
        mu (float or np.ndarray): Shear modulus.
        kappa (float or np.ndarray): Bulk modulus.
    """
    mu = E / (2 * (1 + nu))  # Shear modulus (mu)
    kappa = E / (3 * (1 - 2 * nu))  # Bulk modulus (kappa)
    return kappa, mu

def kappa_mu_to_E_nu(kappa, mu):
    """
    Converts bulk modulus (kappa) and shear modulus (mu) to Young's modulus (E) and Poisson's ratio (nu).

    Parameters:
        kappa (float or np.ndarray): Bulk modulus.
        mu (float or np.ndarray): Shear modulus.
    
    Returns:
        E (float or np.ndarray): Young's modulus.
        nu (float or np.ndarray): Poisson's ratio.
    """
    E = 9 * kappa * mu / (3 * kappa + mu)  # Young's modulus (E)
    nu = (3 * kappa - 2 * mu) / (2 * (3 * kappa + mu))  # Poisson's ratio (nu)
    return E, nu

def calculate_radius(volume_fraction, num_particles):
    return ((volume_fraction*3*np.sqrt(np.pi)/4)/(num_particles*np.pi**(DIMENSION/2)))**(1/DIMENSION)










