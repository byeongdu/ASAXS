import numpy as np
import xraydb
import sys
import matplotlib.pyplot as plt

# Function to load numeric data from an ASCII text file
def load_ascii_data(file_path):
    # data format : AtomicN, E_1, E_2, E_3, ... (first row)
    #            : q1, Iq1_1, Iq1_2, Iq1_3, ... (subsequent rows)
    # where q is the independent variable and E_i are the energy values
    # and Iq1_i are the measured intensities at ith energy (intensity values)
    try:
        data = np.loadtxt(file_path)
        q = data[1:, 0]  # First column as independent variable
        element = data[0, 0]  # First element in the first row (assumed to be the element name)
        energies = data[0, 1:]  # X-ray energies from the first row
        data = data[1:, 1:]  # Measured intensities (Iq) from the subsequent rows
        return q, data, int(element), energies
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None, None

# Function to compute f1 and f2 for given x-ray energies
def compute_f1_f2(energies, element):
    # energies should be in keV
    fprime = []
    fdprime = []
    for energy in energies:
        fp = xraydb.f1_chantler(element, energy*1000)  # Convert keV to eV
        fdp = xraydb.f2_chantler(element, energy*1000)  # Convert keV to eV
        fprime.append(fp)
        fdprime.append(fdp)
    return np.array(fprime), np.array(fdprime)

def create_mx3_array(energies, fp, fdp):
    m = len(energies)
    array = np.zeros((m, 3))
    array[:, 0] = 1
    array[:, 1] = 2 * fp
    array[:, 2] = (fp**2 + fdp**2)
    return array

def compute_Iq(q, Im, energies, element):
    # Compute Iq using the matrix A and the q values
    # This is a placeholder function. The actual computation will depend on the specific model used.
    # For now, we will just return a dummy array of the same shape as A
    # when the element is given as an atomic number, find the element name
    xraydb_element = xraydb.atomic_symbol(element)
    print(f"Element: {xraydb_element}")
    # Compute f1 and f2 for the given element and energies
    # energies should be in keV
    fp, fdp = compute_f1_f2(energies, xraydb_element)
    # Plot f1 (fp) and f2 (fdp) vs energy
    plt.figure()
    plt.plot(energies, fp, label="f1 (fp)")
    plt.plot(energies, fdp, label="f2 (fdp)")
    plt.xlabel("Energy (keV)")
    plt.ylabel("f1, f2")
    plt.title(f"f1 and f2 vs Energy for {xraydb_element}")
    plt.legend()
    plt.grid()
    plt.show()
    # Create the matrix A using the computed f1 and f2 values
    # and the energies
    A = create_mx3_array(energies, fp=fp, fdp   =fdp)
    Iq = np.linalg.lstsq(A, Im.T, rcond=None)[0]  # Solve for Iq using least squares
    Iq = np.array(Iq).T  # Transpose to match the expected output shape
    # Plot Iq vs q for all three columns
    plt.figure(figsize=(12, 6))

    # Left subplot: q vs Im
    plt.subplot(1, 2, 1)
    for i in range(Im.shape[1]):
        plt.loglog(q, Im[:, i], label=f"Im_{energies[i]:.3f} keV")
    plt.xlabel("q")
    plt.ylabel("Im")
    plt.title("q vs Im")
    plt.legend()
    plt.grid()

    # Right subplot: q vs Iq
    plt.subplot(1, 2, 2)
    plt.loglog(q, Iq[:, 0], label="Iq0")
    plt.loglog(q, Iq[:, 1], label="Iq_cross")
    plt.loglog(q, Iq[:, 2], label="Iq_resonant")
    plt.xlabel("q")
    plt.ylabel("Iq")
    plt.title("q vs Iq")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return Iq, A

def save_results(file_path, A, q, Iq):
    # Save the matrix A to a new file
    output_file_A = file_path.rsplit(".", 1)[0] + "_A.txt"
    np.savetxt(output_file_A, A, header="% Matrix A (3 columns: 1, 2*f1, f1^2 + f2^2)", comments='')
    print(f"Matrix A saved to {output_file_A}")
    # Save the results to a new file
    output_file = file_path.rsplit(".", 1)[0] + "_Iq.txt"
    with open(output_file, 'w') as f:
        f.write("% q Iq0 Iq_cross Iq_resonant\n")
        for qi, iq in zip(q, Iq):
            f.write(f"{qi} {iq[0]} {iq[1]} {iq[2]}\n")
    print(f"Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python asaxs.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    q, Im, element, energies = load_ascii_data(file_path)
    if q is not None and energies is not None:
        Iq, A = compute_Iq(q, Im, energies, element)  # Example element
        save_results(file_path, A, q, Iq)