from math import sqrt
from math import exp
from math import sin
from math import cos
from math import isnan
from math import sinh
from math import cosh
from math import degrees
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from matplotlib.pyplot import MultipleLocator

import time
import cmath
import numpy as np
import matplotlib.pyplot as plt


def riccati_bessel_first(order, argument):
    '''
    Riccati bessel function of the first kind.

    Parameters
    ----------
    order : Int
        DESCRIPTION. A real or complex value, in this case it's passed
                     a real
    argument : Float
        DESCRIPTION. Value of the Riccatci bessel function of the first kind

    Returns
    -------
    TYPE : Float
        DESCRIPTION. Value of the Riccatci bessel function of the first kind

    '''
    
    return argument * spherical_jn(order, argument)


def riccati_bessel_second(order, argument):
    '''
    Riccati bessel function of the second kind.

    Parameters
    ----------
    order : Int
        DESCRIPTION. A real or complex value, in this case it's passed
                     a real
    argument : Float
        DESCRIPTION. Value of the Riccatci bessel function of the second kind

    Returns
    -------
    TYPE : Float
        DESCRIPTION. Value of the Riccatci bessel function of the second kind

    '''
    
    return argument * spherical_yn(order, argument)


def get_optical_potential(radius, order):
    '''
    According formulas to calculate the value of Saxon Woods potential.

    Parameters
    ----------
    radius : Float
        DESCRIPTION. The radius away from the target
    order : Int
        DESCRIPTION. Partial wave value

    Returns
    -------
    TYPE : Complex
        DESCRIPTION. Complex value of Saxon Woods potential

    '''

    if order == 0:
        potreal = (150.0 / (1.0 + exp((radius - 1.65) / 0.10))) \
                - (9.2 / (1.0 + exp((radius - 3.72) / 0.40)))
        potimag = - 5.0 / (1.0 + exp((radius - 3.72)/ 0.40))
    elif order == 2:        
        potreal = (150.0 / (1.0 + exp((radius - 1.63) / 0.05))) \
                    - (16.0 / (1.0 + exp((radius - 3.55) / 0.3)))
        potimag = - 5.0 / (1.0 + exp((radius - 3.55)/ 0.3))
    elif order == 4:
        potreal = (220.0 / (1.0 + exp((radius - 1.20) / 0.05))) \
                    - (71.0 / (1.0 + exp((radius - 2.48) / 0.46)))
        potimag = - 5.0 / (1.0 + exp((radius - 2.48)/ 0.46)) 
    else:
        print('Order number error, please enter the order number equal to 0, 2 or 4')
    
    #unit conversion
    real = potreal / 10.367
    imag = potimag / 10.367   
    
    return complex(real, imag)


def get_phases(potential_value, radius, energy, order, previous_phase_e,
               previous_phase_n, fun_type):  
    '''
    Given the previous values of the phases, calculate the new phases.

    Parameters
    ----------
    potential_value : Float
        DESCRIPTION. Complex potential value of Saxon Woods potential
    radius : Float
        DESCRIPTION. The radius away from the target
    energy : Float
        DESCRIPTION. The energy of the scattering electron
    order : Int
        DESCRIPTION. Partial wave value
    previous_phase_e : Float
        DESCRIPTION. The previous real phase
    previous_phase_n : Float
        DESCRIPTION. The previous imag phase
    fun_type : String
        DESCRIPTION. Type of the formula, equal to 'single' or 'complex'

    Returns
    -------
    TYPE: Float
        DESCRIPTION. The new real phase
    TYPE: Float
        DESCRIPTION. The new imag phase

    '''
    #Come from Jana'1992
    if fun_type == 'single':
        rb_first = riccati_bessel_first(order, radius * energy)
        rb_second = riccati_bessel_second(order, radius * energy)   
        prefactor = - 1 /  energy
               
        cos_phase = cmath.cos(complex(previous_phase_e, previous_phase_n))
        sin_phase = cmath.sin(complex(previous_phase_e, previous_phase_n))
        sl_e = cos_phase * rb_first - sin_phase * rb_second    
        
        new_phase_e = prefactor * potential_value * sl_e * sl_e
        
        return new_phase_e.real, new_phase_e.imag
    
    #Come from Jain'1991. I fixed some typos
    elif fun_type == 'coupled':
        rb_first = riccati_bessel_first(order, radius * energy)
        rb_second = riccati_bessel_second(order, radius * energy)

        X = cosh(previous_phase_n) * (rb_second * sin(previous_phase_e) - rb_first * cos(previous_phase_e))
        Y = sinh(previous_phase_n) * (rb_second * cos(previous_phase_e) + rb_first * sin(previous_phase_e))

        dy_0 = -(1.0/energy) * (potential_value.real * (X ** 2 - Y ** 2) - 2 * potential_value.imag * X * Y)   
        dy_1 = - (1.0/energy) * (2 * potential_value.real * X * Y + potential_value.imag * (X ** 2 - Y ** 2))
    
        return dy_0, dy_1 
    
    else:
        print('function type error, please enter the function type equal to single or complex')
    

def get_next_e_next_n_rk45(radius, energy, order, e_value, n_value, step, fun_type):
    '''
    The formula of Runge-Kutta method, which is used to calculate next phase shift value.

    Parameters
    ----------
    radius : Float
        DESCRIPTION. The radius away from the target
    energy : Float
        DESCRIPTION. The energy of the scattering electron
    order : Int
        DESCRIPTION. Partial wave value
    e_value : Float
        DESCRIPTION. The previous real phase shift
    n_value : Float
        DESCRIPTION. The previous imag phase shift
    step : Float
        DESCRIPTION. The step of Runge-Kutta method
    fun_type : String
        DESCRIPTION. Type of the function of get_phases, equal to 'single' or 'complex'

    Returns
    -------
    next_e : Float
        DESCRIPTION. The new real phase shift
    next_n : Float
        DESCRIPTION. The new imag phase shift

    '''
   
    k_list = []
    m_list = []
    potential_value = []
    r_list = [radius,
              radius + step / 2.0,
              radius + step / 2.0,
              radius + step]
    for rvalue in r_list:
        potential_value.append(get_optical_potential(rvalue, order))
    # First set
    new_phase_e, new_phase_n = get_phases(potential_value[0], r_list[0],
                                          energy, order,
                                          e_value, n_value, fun_type)
    k_list.append(new_phase_e * step)
    m_list.append(new_phase_n * step)
    # Second set
    new_phase_e, new_phase_n = get_phases(potential_value[1], r_list[1],
                                          energy, order,
                                          e_value + k_list[0] / 2.0,
                                          n_value + m_list[0] / 2.0, fun_type)
    k_list.append(new_phase_e * step)
    m_list.append(new_phase_n * step)
    # Third set
    new_phase_e, new_phase_n = get_phases(potential_value[2], r_list[2],
                                          energy, order,
                                          e_value + k_list[1] / 2.0,
                                          n_value + m_list[1] / 2.0, fun_type)
    k_list.append(new_phase_e * step)
    m_list.append(new_phase_n * step)
    # Fourth set
    new_phase_e, new_phase_n = get_phases(potential_value[3], r_list[3],
                                          energy, order,
                                          e_value + k_list[2],
                                          n_value + m_list[2], fun_type)
    k_list.append(new_phase_e * step)
    m_list.append(new_phase_n * step)
    
    next_e = e_value + k_list[0] / 6.0 + k_list[1] / 3.0\
            + k_list[2] / 3.0 + k_list[3] / 6.0
    next_n = n_value + m_list[0] / 6.0 + m_list[1] / 3.0\
            + m_list[2] / 3.0 + m_list[3] / 6.0 
    
    return next_e, next_n


def integrate_function(energy, order, iternumber, fun_type):
    '''
    Integration over Saxon Woods potential.

    Parameters
    ----------
    energy : Float
        DESCRIPTION. The energy of the scattering electron
    order : Int
        DESCRIPTION. The order of the bessel function and Saxon Woods potential
    iternumber : Int
        DESCRIPTION. Iteration number of Runge-Kutta method
    fun_type : String
        DESCRIPTION. Type of the function of get_phases, equal to 'single' or 'complex'

    Returns
    -------
    TYPE : Float, Float
        DESCRIPTION. Real and imag parts of phase shift

    '''
    
    rmin = 0.01
    rmax = 8.0

    rstep = (rmax - rmin) / iternumber
    e_values = []
    n_values = []
    e_values.append(0.0)
    n_values.append(0.0)
    
    #unit conversion
    energy_new = energy / 10.367
    
    #E = k*k
    k_energy = sqrt(energy_new)
    for i in range(0, iternumber):
        radius = rmin + i * rstep
        next_e, next_n= get_next_e_next_n_rk45(radius, k_energy, order, e_values[i], n_values[i],
                                          rstep, fun_type)
        if isnan(next_e) or isnan(next_n):
            break
        else:
            e_values.append(next_e)
            n_values.append(next_n)
            
    return degrees(e_values[-1]), degrees(n_values[-1])

    
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
    fun_type : String
        DESCRIPTION. Type of the function of get_phases, equal to 'single' or 'coupled'

    Returns
    -------
    None.

    '''
    
    print('order = ')
    print(order)
    print('iternumber = ')
    print(iternumber)
    print('fun_type = ')
    print(fun_type)   
    
    #Get E, which equal to k * k
    energy_grid = np.linspace(minenergy, 50, num=25, endpoint=False)   
        
    e = []
    n = []
    
    start = time.perf_counter()
    for energy in energy_grid:
        phase_e, phase_n = integrate_function(energy, order, iternumber, fun_type)
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

    plt.plot(energy_grid, e, label = 'real; This Work', color='b')
    plt.plot(energy_grid, n, label = 'image', color='r')
    plt.xlabel('Ecm(MeV)')
    plt.ylabel('phase shift(degrees)')
    x_major_locator = MultipleLocator(4)
    y_major_locator = MultipleLocator(20)    
    ax=plt.gca()   
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)    
    plt.legend()
    plt.show()


def plot_point_phase(energy, order, fun_type):
    '''
    Plot the phase shift for a single point.

    Parameters
    ----------
    energy : Float
        DESCRIPTION. The energy of the scattering electron (MeV)
    order : Int
        DESCRIPTION. The order of the bessel function and Saxon Woods potential,
        equal to 0, 2 or 4
    fun_type : String
        DESCRIPTION. Type of the function of get_phases, equal to 'single' or 'coupled'

    Returns
    -------
    None.

    '''
    
    #rmin value dose not seem to affect the results
    rmin = 0.01
    rmax = 10.0

    rstep = (rmax - rmin) / 1000.0
    e_values = []
    n_values = []
    r_values = []
    e_values.append(0.0)
    n_values.append(0.0)
    r_values.append(0.0)
    
    #unit conversion
    energy_new = energy / 10.367
    
    #E = k*k
    k_energy = sqrt(energy_new)
    for i in range(0, 1000):
        radius = rmin + i * rstep
        r_values.append(radius)   
        next_e, next_n= get_next_e_next_n_rk45(radius, k_energy, order, e_values[i], n_values[i],
                                          rstep, fun_type)
        if isnan(next_e) or isnan(next_n):
            break
        else:
            e_values.append(next_e)
            n_values.append(next_n)
            
    for j in range(0, 1001):
        e_values[j] = degrees(e_values[j])
        n_values[j] = degrees(n_values[j])
   
    plt.xlabel('radius(fm)')
    plt.ylabel('phase shift(degrees)')
    plt.title('Partial wave l=4; Ecm=12MeV') 
    plt.plot(r_values, e_values, label = 'real; This Work', color='b')
    plt.plot(r_values, n_values, label = 'imag', color='r')
    plt.legend()
    plt.show()

#Plot phase shifts of Saxon Woods potential
#We can change the itertion number to test the calculate speed
#We can compare the speed of different formulas by changing the string to 'single' or 'coupled'
print('Plot phase shifts of Saxon Woods potential')
plot_saxon_woods(0, 2, 1000, 'single')
plot_saxon_woods(2, 2, 1000, 'single')
plot_saxon_woods(4, 4, 1000, 'single')

#Plot the phase shift for a single point
print('Plot the phase shift for a single point')
plot_point_phase(12, 4, 'single')

