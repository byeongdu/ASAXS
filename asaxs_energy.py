import sys
import numpy as np
import xraydb
import argparse

def get_absorption_edge(element, min_energy_keV=6, max_energy_keV=34):
    k_edge = xraydb.xray_edge(element, 'K')
    l3_edge = xraydb.xray_edge(element, 'L3')

    chosen_edge = None
    edge_type = None
    print(f"Element: {element}, K-edge: {k_edge}, L3-edge: {l3_edge}")
    if k_edge and (min_energy_keV <= k_edge.energy/1000 <= max_energy_keV):
        chosen_edge = k_edge.energy
        edge_type = 'K'
    elif l3_edge and (min_energy_keV <= l3_edge.energy/1000 <= max_energy_keV):
        chosen_edge = l3_edge.energy
        edge_type = 'L3'

    return chosen_edge, edge_type

def get_f1_curve(element, edge_energy_eV, energy_range_eV=200, num_points=300):
    energies = np.linspace(edge_energy_eV - energy_range_eV, edge_energy_eV, num_points)
    f1_values = np.array([xraydb.f1_chantler(element, en) for en in energies])
    return energies, f1_values

def select_uniform_f1_points(energies, f1_values, num_points=15):
    sorted_idx = np.argsort(f1_values)
    f1_sorted = f1_values[sorted_idx]
    energy_sorted = energies[sorted_idx]

    f1_min, f1_max = f1_sorted[0], f1_sorted[-1]
    f1_targets = np.linspace(f1_min, f1_max, num_points)

    selected_energies = []
    selected_f1 = []

    for target in f1_targets:
        idx = np.abs(f1_sorted - target).argmin()
        selected_energies.append(energy_sorted[idx])
        selected_f1.append(f1_sorted[idx])

    # sort back by energy
    selected = sorted(zip(selected_energies, selected_f1), key=lambda x: x[0])
    energies_out, f1_out = zip(*selected)

    return np.array(energies_out), np.array(f1_out)

def choose_asaxs(element, energy_range_eV=1000, num_points=20):
    edge, edgetype = get_absorption_edge(element)
    if edge is None:
        raise ValueError(f"No suitable K or L3 edge found for {element} between 6 and 34 keV.")
    energies, f1_values = get_f1_curve(element, edge, energy_range_eV, num_points)
    selected_energies, selected_f1 = select_uniform_f1_points(energies, f1_values, num_points)
    np.savetxt(f"{element}_selected_energies.txt", selected_energies, fmt="%.3f")
    return selected_energies, selected_f1, edge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select ASAXS energies for an element.")
    parser.add_argument("element", type=str, help="Element symbol (e.g., Fe, Cu)")
    parser.add_argument("--energy_range_eV", type=float, default=1000, help="Energy range in eV (default: 1000)")
    parser.add_argument("--num_points", type=int, default=20, help="Number of points (default: 20)")
    args = parser.parse_args()

    selected_energies, selected_f1, edge = choose_asaxs(args.element, args.energy_range_eV, args.num_points)
    print(f"Selected energies (eV): {selected_energies}")
    print(f"Edge energy (eV): {edge}")