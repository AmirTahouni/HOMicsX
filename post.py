import numpy as np

from config import *
from utils import *

#-------------------------------------------------| post |-------------------------------------------------#
def post_process(C_hom, E_particle, E_matrix, nu_particle, nu_matrix, volume_fraction):
    # Print the final stiffness matrix
    print("\n" + "="*90)
    print("|" + "-"*88 + "|")
    print("|       Homogenized Tensor Of Elasticity (C_hom) - Final Results        |")
    print("|" + "-"*88 + "|")
    print("="*90 + "\n")
    with np.printoptions(suppress=True, precision=3):
        print(C_hom)
    print("\n" + "="*90)

    # Extract the effective moduli from C_hom
    def extract_effective_moduli(C_hom):
        """
        Extracts effective moduli (kappa_eff, mu_eff, E_eff, nu_eff) from the homogenized stiffness matrix.
        """
        # Extract the relevant entries:
        C11, C22, C33 = C_hom[0, 0], C_hom[1, 1], C_hom[2, 2]
        C12, C13, C23 = C_hom[0, 1], C_hom[0, 2], C_hom[1, 2]
        C44, C55, C66 = C_hom[3, 3], C_hom[4, 4], C_hom[5, 5]

        # Effective lame moduli:
        lambda_eff = C12
        mu_eff = C44

        # Effective bulk modulus
        kappa_eff = lambda_eff + 2/3 * mu_eff

        # Convert to effective Young's modulus and Poisson's ratio:
        E_eff, nu_eff = kappa_mu_to_E_nu(kappa_eff, mu_eff)

        return kappa_eff, mu_eff, E_eff, nu_eff

    kappa_res, mu_res, E_res, nu_res = extract_effective_moduli(C_hom)
    print(f"\nEffective Moduli from Homogenized Tensor:")
    print(f"  E_eff: {E_res:.3e} Pa")
    print(f"  nu_eff: {nu_res:.3f}")
    print(f"  kappa_eff: {kappa_res:.3e} Pa")
    print(f"  mu_eff: {mu_res:.3e} Pa\n")

    def WILLIS_dilute(E_matrix, nu_matrix, E_particle, nu_particle, volume_fraction):
        kappa, mu = E_nu_to_kappa_mu(E_matrix, nu_matrix)
        kappap, mup = E_nu_to_kappa_mu(E_particle, nu_particle)

        kappaWil = kappa + volume_fraction * (kappap - kappa) * (3 * kappa + 4 * mu) / (3 * kappap + 4 * mu)
        muWil = mu + volume_fraction * 5 * mu * (mup - mu) * (3 * kappa + 4 * mu) / (mu * (9 * kappa + 8 * mu) + 6 * mup * (kappa + 2 * mu))

        E_eff, nu_eff = kappa_mu_to_E_nu(kappaWil, muWil)

        print("\n" + "="*90)
        print("|" + "-"*88 + "|")
        print("|       Willis Dilute Approximation Results                        |")
        print("|" + "-"*88 + "|")
        print("="*90 + "\n")
        print(f"  E_eff (Willis dilute): {E_eff:.3e} Pa")
        print(f"  nu_eff (Willis dilute): {nu_eff:.3f}")
        print(f"  kappa_eff (Willis dilute): {kappaWil:.3e} Pa")
        print(f"  mu_eff (Willis dilute): {muWil:.3e} Pa\n")

        return E_eff, nu_eff

    E_dil, nu_dil = WILLIS_dilute(E_matrix, nu_matrix, E_particle, nu_particle, volume_fraction)
    kappa_dil, mu_dil = E_nu_to_kappa_mu(E_dil, nu_dil)

    def WILLIS_SC(E_matrix, nu_matrix, E_particle, nu_particle, volume_fraction):
        kappa, mu = E_nu_to_kappa_mu(E_matrix, nu_matrix)
        kappap, mup = E_nu_to_kappa_mu(E_particle, nu_particle)
        c = volume_fraction

        def calculate_effective_moduli(kappa1, kappa2, mu1, mu2, c1, tol=1e-6, max_iter=10000):
            kappa_eff = kappa2
            mu_eff = mu2

            for iteration in range(max_iter):
                kappa_prev = kappa_eff
                mu_prev = mu_eff

                kappa_eff = kappa2 + c1 * ((kappa1 - kappa2) * (3 * kappa_prev + 4 * mu_prev) /
                                        (3 * kappa1 + 4 * mu_prev))
                mu_eff = mu2 + c1 * (5 * (mu1 - mu2) * mu_prev * (3 * kappa_prev + 4 * mu_prev) /
                                    (mu_prev * (9 * kappa_prev + 8 * mu_prev) + 6 * mu1 * (kappa_prev + 2 * mu_prev)))

                if abs(kappa_eff - kappa_prev) < tol and abs(mu_eff - mu_prev) < tol:
                    break

            return kappa_eff, mu_eff

        kappa0, mu0 = calculate_effective_moduli(kappap, kappa, mup, mu, c)

        E_eff, nu_eff = kappa_mu_to_E_nu(kappa0, mu0)

        print("\n" + "="*90)
        print("|" + "-"*88 + "|")
        print("|       Willis Self-Consistent Approximation Results               |")
        print("|" + "-"*88 + "|")
        print("="*90 + "\n")
        print(f"  E_eff (Willis SC): {E_eff:.3e} Pa")
        print(f"  nu_eff (Willis SC): {nu_eff:.3f}")
        print(f"  kappa_eff (Willis SC): {kappa0:.3e} Pa")
        print(f"  mu_eff (Willis SC): {mu0:.3e} Pa\n")

        return E_eff, nu_eff

    E_sc, nu_sc = WILLIS_SC(E_matrix, nu_matrix, E_particle, nu_particle, volume_fraction)
    kappa_sc, mu_sc = E_nu_to_kappa_mu(E_sc, nu_sc)

    def hs_bulk_bounds(f_particle, E_particle, nu_particle, E_matrix, nu_matrix):
        kappa_matrix, mu_matrix = E_nu_to_kappa_mu(E_matrix, nu_matrix)
        kappa_particle, mu_particle = E_nu_to_kappa_mu(E_particle, nu_particle)

        f_matrix = 1.0 - f_particle
        
        if kappa_particle == kappa_matrix:
            raise ValueError("kappa_particle must differ from kappa_matrix for nontrivial bounds.")
        
        term_common_lower = 1.0 / (kappa_particle - kappa_matrix)
        term_common_lower += f_matrix / (kappa_matrix + (4.0/3.0)*mu_matrix)
        kappa_lower = kappa_matrix + f_particle / term_common_lower

        term_common_upper = 1.0 / (kappa_particle - kappa_matrix)
        term_common_upper += f_particle / (kappa_particle + (4.0/3.0)*mu_matrix)
        kappa_upper = kappa_particle - f_matrix / term_common_upper

        return kappa_lower, kappa_upper

    def hs_shear_bounds(f_particle, E_particle, nu_particle, E_matrix, nu_matrix):
        kappa_matrix, mu_matrix = E_nu_to_kappa_mu(E_matrix, nu_matrix)
        kappa_particle, mu_particle = E_nu_to_kappa_mu(E_particle, nu_particle)

        f_matrix = 1.0 - f_particle
        
        if mu_particle == mu_matrix:
            raise ValueError("mu_particle must differ from mu_matrix for nontrivial bounds.")
        
        denom_lower = 1.0 / (mu_particle - mu_matrix)
        denom_lower += (2.0 * f_matrix * (kappa_matrix + 2.0 * mu_matrix)) / (5.0 * mu_matrix * (3.0*kappa_matrix + 4.0*mu_matrix))
        mu_lower = mu_matrix + f_particle / denom_lower

        denom_upper = 1.0 / (mu_particle - mu_matrix)
        denom_upper += (2.0 * f_particle * (kappa_matrix + 2.0 * mu_matrix)) / (5.0 * mu_matrix * (3.0*kappa_matrix + 4.0*mu_matrix))
        mu_upper = mu_particle - f_matrix / denom_upper

        return mu_lower, mu_upper

    kappa_lower, kappa_upper = hs_bulk_bounds(volume_fraction, E_particle, nu_particle, E_matrix, nu_matrix)
    mu_lower, mu_upper = hs_shear_bounds(volume_fraction, E_particle, nu_particle, E_matrix, nu_matrix)

    print("\n" + "="*90)
    print("|" + "-"*88 + "|")
    print("|       Hashin–Shtrikman Bounds                                      |")
    print("|" + "-"*88 + "|")
    print("="*90 + "\n")
    print(f"  kappa_lower: {kappa_lower:.3e} Pa")
    print(f"  kappa_upper: {kappa_upper:.3e} Pa")
    print(f"  mu_lower: {mu_lower:.3e} Pa")
    print(f"  mu_upper: {mu_upper:.3e} Pa\n")

    output = {
        "Result": {
            "E_eff": E_res,
            "nu_eff": nu_res,
            "kappa_eff": kappa_res,
            "mu_eff": mu_res
        },
        "WILLIS-Dilute": {
            "E_eff": E_dil,
            "nu_eff": nu_dil,
            "kappa_eff": kappa_dil,
            "mu_eff": mu_dil
        },
        "WILLIS-SelfConsistant": {
            "E_eff": E_sc,
            "nu_eff": nu_sc,
            "kappa_eff": kappa_sc,
            "mu_eff": mu_sc
        },
        "Hashin-Strikman": {
            "kappa_lower": kappa_lower,
            "kappa_upper": kappa_upper,
            "mu_lower": mu_lower,
            "mu_upper": mu_upper
        }
    }

    return output
