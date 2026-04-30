import numpy as np
import xraydb
import sys
import pyqtgraph as pg

_windows = []
from scipy.optimize import minimize
from lmfit import Parameters 
from lmfit import minimize as lmfit_minimize
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
    # f = f0 + fp + i*fdp; see https://www.xtal.iqf.csic.es/Cristalografia/parte_05_7-en.html
    # f0 = xraydb.f0(element, energy*1000)  # Convert keV to eV
    # fprime = xraydb.f1_chantler(element, energy*1000)  # Convert keV to eV  
    # fdprime = xraydb.f2_chantler(element, energy*1000)  # Convert keV to eV
    # f1 = f0 + fprime
    # f2 = fdprime  
    f1 = []
    f2 = []
    for energy in energies:
        f0 = xraydb.f0(element,0)  # Convert keV to eV
        #fp = f0[0] + xraydb.f1_chantler(element, energy*1000)  # Convert keV to eV
        fp = xraydb.f1_chantler(element, energy*1000)  # Convert keV to eV
        fdp = xraydb.f2_chantler(element, energy*1000)  # Convert keV to eV
        f1.append(fp)
        f2.append(fdp)
    return np.array(f1), np.array(f2)

def create_mx3_array(energies, fp, fdp):
    m = len(energies)
    array = np.zeros((m, 3))
    array[:, 0] = 1
    array[:, 1] = 2 * fp
    array[:, 2] = (fp**2 + fdp**2)
    return array

def objective_function(x, A, Im):
    # x should be Iq0, Iq_cross, Iq_resonant
    #print(x)
    #print(A[0])
    #print(Im.T)
    #print(A[0][:,0]*x[0] + 2*A[0][:,1]*x[1] + A[0][:,2])
    # 1*Iq0 + 2*fp*Iq_cross + (fp^2 + fdp^2)*Iq_resonant
    return np.sum(np.abs(Im.T - (A[:,0]*x[0] + 2*A[:,1]*x[1] + A[:,2]*x[2]))**2)
    #return np.sum(np.array([np.sum([A[i,j]*x[j]-Im[i] for j in range(len(x))]) for i in range(A.shape[0])])**2)

def lmfit_finderrbars_new(x, fp, fpp, I, Ierr):
    params=Parameters()
    params.add('Io',value=x[0],min=x[0]/1E3,max=x[0]*1E3,vary=True)
    params.add('Ir',value=x[1],min=x[1]/1E8,max=x[0]*1E3,vary=True)
    params.add('alf',value=x[2],min=0,max=np.pi,vary=True)
    if I[0] ==0:
        return 1, 1, 1, 1, 1, 1, 1
    result=lmfit_minimize(residual_new, params, args=(fp,fpp,I, Ierr))
    rpars=result.params
    return rpars['Io'].value, rpars['Io'].stderr, rpars['Ir'].value, rpars['Ir'].stderr, rpars['alf'].value, rpars['alf'].stderr, result.redchi

def residual_new(param, fp, fpp, I, Ierr):
    Io, Ir, alf = param['Io'].value, param['Ir'].value, param['alf'].value
    #df = np.sum(((I-sturhman(Io, Ir, alf, fp, fpp))/Ierr)**2)/3.0 # chi squared.
    #print(df, Ierr)
    return (I-sturhman(Io, Ir, alf, fp, fpp))/Ierr

def sturhman(Io, Ir, alf, fp, fpp):
    return Io + (fp**2+fpp**2)*Ir + 2*np.sqrt(np.abs(Io)*np.abs(Ir))*fp*np.cos(alf)

def fit(A, Im):
    # Fit the data using least squares
    # A is the design matrix and Im is the measured intensities
    # We will use a regularization term to avoid overfitting
    lambda_reg = 1e-6  # small regularization factor
    A_reg = A.T @ A + lambda_reg * np.identity(A.shape[1])
    b_reg = A.T @ Im.T

    Iq = np.linalg.solve(A_reg, b_reg)
    residuals = np.linalg.norm(A @ Iq - Im.T, axis=0)  # Compute residuals for each energy
    rank = np.linalg.matrix_rank(A)
    s = np.linalg.svd(A, compute_uv=False)
    return Iq, residuals, rank, s

def fit2(fp, dfp, Im):
    Iq = []
    tot = []
    Ierr = []
    sh = Im.shape
    Io = Im[0][0]
    Ir = Im[0][0]/1E4
    alf = 0.01
    for i in range(sh[0]):
        if i==0:
            Ncycle = [0,1,2]
        else:
            Ncycle = [1]
        for k in Ncycle:
            xv = [Io, Ir, alf]
            Io, Ioerr, Ir, Irerr, alf, alferr, redchi1 = lmfit_finderrbars_new(xv, fp, dfp, Im[i,:], np.sqrt(Im[i,:]))
        Ic = 2*np.sqrt(Io*Ir)*fp[0]*np.cos(alf)
        if Ioerr is None:
            Ioerr=0.1*Io
        if Irerr is None:
            Irerr=0.1*Ir
        if alferr is None:
            alferr=0.1*alf
        Icerr = np.sqrt(Ioerr ** 2 / Io + Irerr ** 2 / Ir + alferr**2)
        x1, x1err, x2, x2err, x3, x3err = Io, Ioerr, Ic, Icerr, Ir, Irerr
        total_new=Io+(fp[0]**2+dfp[0]**2)*Ir+Ic
        Iq.append([x1, x2, x3])
        tot.append(total_new)
        Ierr.append([x1err, x2err, x3err])
    return np.array(Iq), np.array(tot), np.array(Ierr)

def fit3(A, Im):
    cons=({'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[2]},)
    Iq = []
    sh = Im.shape
    for i in range(sh[0]):
        res=minimize(objective_function,[0.0,0.0,0.0],args=(A,Im[i,:]),
                     constraints=cons,bounds=((0,None),(None,None),(0,None)))
        #print(res)
        Iq.append(res.x)
    return np.array(Iq)

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
    #print(f"f1 (fp): {fp}")
    #print(f"f2 (fdp): {fdp}")
    # Plot f1 (fp) and f2 (fdp) vs energy
    win1 = pg.GraphicsLayoutWidget(title=f"f1 and f2 vs Energy for {xraydb_element}", show=True)
    _windows.append(win1)
    p1 = win1.addPlot(title=f"f1 and f2 vs Energy for {xraydb_element}")
    p1.addLegend()
    p1.plot(energies, fp, pen='b', name="f1 (fp)")
    p1.plot(energies, fdp, pen='r', name="f2 (fdp)")
    p1.setLabel('bottom', "Energy (keV)")
    p1.setLabel('left', "f1, f2")
    p1.showGrid(x=True, y=True)
    # Create the matrix A using the computed f1 and f2 values
    # and the energies
    A = create_mx3_array(energies, fp=fp, fdp   =fdp)
    # Solve for Iq using least squares
    #Iq, residuals, rank, s = np.linalg.lstsq(A, Im.T, rcond=None)
    Iq, tot, Ierr = fit2(fp, fdp, Im)
    residuals = Ierr
    #residuals = np.linalg.norm(A @ Iq - Im.T, axis=0)  # Compute residuals for each energy
    #rank = np.linalg.matrix_rank(A)
    #s = np.linalg.svd(A, compute_uv=False)
    #print(f"Rank of A: {rank}")
    #print(f"Singular values of A: {s}")
    #print(f"Fit residuals: {residuals}")
    #print(f"A shape: {A.shape[0]}")
    #Iq = np.array(Iq).T  # Transpose to match the expected output shape
    #residuals = np.array(residuals).T  # Transpose to match the expected output shape
    # Plot Iq vs q for all three columns
    win2 = pg.GraphicsLayoutWidget(title="ASAXS Results", show=True)
    _windows.append(win2)

    # Left plot: q vs Im
    p2 = win2.addPlot(title="Measured Intensities")
    p2.setLogMode(x=True, y=True)
    p2.setLabel('bottom', "q")
    p2.setLabel('left', "I(q)_m")
    p2.addLegend()
    p2.showGrid(x=True, y=True)
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'w']
    for i in range(Im.shape[1]):
        p2.plot(q, Im[:, i], pen=colors[i % len(colors)], name=f"Im_{energies[i]:.3f} keV")

    # Right plot: q vs Iq
    p3 = win2.addPlot(title="Three Components of Iq")
    p3.setLogMode(x=True, y=True)
    p3.setLabel('bottom', "q")
    p3.setLabel('left', "Iq")
    p3.addLegend()
    p3.showGrid(x=True, y=True)
    p3.plot(q, Iq[:, 0], pen=None, symbol='o', symbolSize=5, name="Iq0")
    p3.plot(q, Iq[:, 1], pen=None, symbol='o', symbolSize=5, name="Iq_cross")
    p3.plot(q, Iq[:, 2], pen=None, symbol='o', symbolSize=5, name="Iq_resonant")
    return Iq, A, residuals

def save_results(file_path, A, q, Iq, err):
    # Save the matrix A to a new file
    output_file_A = file_path.rsplit(".", 1)[0] + "_A.txt"
    np.savetxt(output_file_A, A, header="% Matrix A (3 columns: 1, 2*f1, f1^2 + f2^2)", comments='')
    print(f"Matrix A saved to {output_file_A}")
    # Save the results to a new file
    output_file = file_path.rsplit(".", 1)[0] + "_Iq.txt"
    with open(output_file, 'w') as f:
        if err.shape[1] == 3:
            f.write("% q Iq0 Iq_cross Iq_resonant Iq0_err Iq_cross_err Iq_resonant_err\n")
            for qi, iq, er in zip(q, Iq, err):
                f.write(f"{qi} {iq[0]} {iq[1]} {iq[2]} {er[0]} {er[1]} {er[2]}\n")
        else:
            f.write("% q Iq0 Iq_cross Iq_resonant Residual\n")
            for qi, iq, er in zip(q, Iq, err):
                f.write(f"{qi} {iq[0]} {iq[1]} {iq[2]} {er}\n")
    print(f"Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    app = pg.mkQApp("ASAXS")
    if len(sys.argv) <= 1:
        print("Usage: python asaxs.py <file_path>")
        print("Usage: python asaxs.py <file_path> <column_starting_to_include>")
        sys.exit(1)
    file_path = sys.argv[1]
    if len(sys.argv)>2:
        omit = int(sys.argv[2])
    else:
        omit = 0
    q, Im, element, energies = load_ascii_data(file_path)
    if q is not None and energies is not None:
        Iq, A, err = compute_Iq(q, Im[:,omit:], energies[omit:], element)  # Example element
        save_results(file_path, A, q, Iq, err)
    pg.exec()