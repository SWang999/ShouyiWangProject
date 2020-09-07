from math import sqrt
from math import degrees
from scipy import integrate

import time
import numpy as np
import matplotlib.pyplot as plt

from tab3_ModelPotential import OpticalPotential

    
def integrate_function_rk45(energy, order, iternumber, optical_potential, fun_type):
    '''
    Integration over the Saxon Woods potential with SciPy.

    Parameters
    ----------
    energy : Float
        DESCRIPTION. The energy of the scattering electron
    order : Int
        DESCRIPTION. The order of the bessel function and Saxon Woods potential
    iternumber : Int
        DESCRIPTION. Iteration number of SciPy
    optical_potential : Class
        DESCRIPTION. Get the value of Saxon Woods potential
    fun_type : String
        DESCRIPTION. Type of the formula, equal to 'single' or 'complex'

    Returns
    -------
    TYPE: Float, Float
        DESCRIPTION. New real and imag parts of phase shift

    '''
    
    rmin = 0.01
    rmax = 8.0
    rstep = (rmax - rmin) / iternumber
    e_values_start = 0.0
    n_values_start = 0.0
    y_vector_start = [e_values_start, n_values_start]
    
    #unit conversion
    energy_new = energy / 10.367    
    #E = k*k
    k_energy = sqrt(energy_new)
    
    optical_potential.update_scattering_energy(k_energy)
    optical_potential.update_order(order)
    if fun_type == 'coupled':
        solver = integrate.RK45(optical_potential.get_phases_coupled, rmin, y_vector_start, rmax, max_step=0.01, atol=1e-5, first_step=rstep)
    elif fun_type == 'single':
        solver = integrate.RK45(optical_potential.get_phases_single, rmin, y_vector_start, rmax, max_step=0.01, atol=1e-5, first_step=rstep)
    
    steps_max = iternumber
    for step in range(steps_max):
        solver.step()
        if solver.status == 'finished':
            break
    
    return degrees(solver.y[0]), degrees(solver.y[1])

    
def plot_saxon_woods(order, minenergy, iternumber, fun_type):
    '''
    Plot phase shifts of Saxon Woods potential.

    Parameters
    ----------
    order : Int
        DESCRIPTION. The order of the bessel function and Saxon Woods potential,
        equal to 0, 2 or 4
    minenergy : Float
        DESCRIPTION. Minimum energy of the potential
    iternumber : Int
        DESCRIPTION. Iteration number of SciPy
    fun_type : String
        DESCRIPTION. fun_type = 'single' or 'coupled'

    Returns
    -------
    None.

    '''

    print('order = ')
    print(order)
    print('iternumber = ')
    print(iternumber)
    
    #Get E, which equal to k * k
    energy_grid = np.linspace(minenergy, 50, num=25, endpoint=False)   
        
    e = []
    n = []
    
    optical_potential = OpticalPotential()
    start = time.perf_counter()
    for energy in energy_grid:
        phase_e, phase_n = integrate_function_rk45(energy, order, iternumber, optical_potential, fun_type)
        e.append(phase_e)
        n.append(phase_n)
    end = time.perf_counter()
    print('run time is: ',end - start)
        
    ax = plt.gca() 
    ax.spines['right'].set_color('none')  
    ax.spines['top'].set_color('none')     
    ax.xaxis.set_ticks_position('bottom')   
    ax.yaxis.set_ticks_position('left')  
    ax.spines['bottom'].set_position(('data', 0)) 
    ax.spines['left'].set_position(('data', 0))

    plt.plot(energy_grid, e, label = 'real')
    plt.plot(energy_grid, n, label = 'image')
    plt.xlabel('Ecm(MeV)')
    plt.ylabel('phase shift(degrees)') 
    plt.legend()
    plt.show()
    
#Calculate phase shifts through SciPy
#We can change the itertion number to test the calculate speed
print('Calculate phase shifts through SciPy')
plot_saxon_woods(0, 2, 1000, 'single')
plot_saxon_woods(2, 2, 1000, 'single')
plot_saxon_woods(4, 4, 1000, 'single')
