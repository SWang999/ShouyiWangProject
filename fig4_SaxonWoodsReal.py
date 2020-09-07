import matplotlib.pyplot as plt
import numpy as np

from math import exp
from math import sin
from math import cos
from math import sqrt
from math import isnan
from math import degrees
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from matplotlib.pyplot import MultipleLocator


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


def get_real_potential(radius, order):
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
    TYPE : Float
        DESCRIPTION. Real value of Saxon Woods potential

    '''
    
    if order == 0:
        potreal = (150.0 / (1.0 + exp((radius - 1.65) / 0.10))) \
                - (9.2 / (1.0 + exp((radius - 3.72) / 0.40)))
    elif order == 2:        
        potreal = (150.0 / (1.0 + exp((radius - 1.63) / 0.05))) \
                    - (16.0 / (1.0 + exp((radius - 3.55) / 0.3)))
    elif order == 4:
        potreal = (220.0 / (1.0 + exp((radius - 1.20) / 0.05))) \
                    - (71 / (1.0 + exp((radius - 2.48) / 0.46)))
    else:
        print('Order number error, please enter the order number equal to 0, 2 or 4')
        
    return potreal 


def get_phases(potential_value, radius, energy, order, previous_phase_e):
    '''
    Given the previous values of the phases, calculate the new phases.

    Parameters
    ----------
    potential_value : Float
        DESCRIPTION. Real potential value of Saxon Woods potential
    radius : Float
        DESCRIPTION. The radius away from the target
    energy : Float
        DESCRIPTION. The energy of the scattering electron
    order : Int
        DESCRIPTION. Partial wave value
    previous_phase_e : Float
        DESCRIPTION. The previous real phase

    Returns
    -------
    new_phase_e : Float
        DESCRIPTION. The new real phase

    '''    
    
    rb_first = riccati_bessel_first(order, radius * energy)
    rb_second = riccati_bessel_second(order, radius * energy)   
    prefactor = - 1 /  energy
    sl_e = cos(previous_phase_e) * rb_first - sin(previous_phase_e) * rb_second    
    real_value = potential_value / 10.367
    
    new_phase_e = prefactor * real_value * sl_e * sl_e
      
    return new_phase_e


def get_next_e(radius, energy, order, e_value, step):
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
    step : Float
        DESCRIPTION. The step of Runge-Kutta method

    Returns
    -------
    next_e : Float
        DESCRIPTION. The new real phase shift

    '''    
   
    k_list = []
    potential_value = []
    r_list = [radius,
              radius + step / 2.0,
              radius + step / 2.0,
              radius + step]
    for rvalue in r_list:
        potential_value.append(get_real_potential(rvalue, order))
    # First set
    new_phase_e = get_phases(potential_value[0], r_list[0],
                             energy, order,
                             e_value)
    k_list.append(new_phase_e * step)
    # Second set
    new_phase_e = get_phases(potential_value[1], r_list[1],
                             energy, order,
                             e_value + k_list[0] / 2.0)
    k_list.append(new_phase_e * step)
    # Third set
    new_phase_e = get_phases(potential_value[2], r_list[2],
                             energy, order,
                             e_value + k_list[1] / 2.0)
    k_list.append(new_phase_e * step)
    # Fourth set
    new_phase_e = get_phases(potential_value[3], r_list[3],
                             energy, order,
                             e_value + k_list[2])
    k_list.append(new_phase_e * step)
    
    next_e = e_value + k_list[0] / 6.0 + k_list[1] / 3.0\
            + k_list[2] / 3.0 + k_list[3] / 6.0

    return next_e


def integrate_function(energy, order):
    '''
    Integration over the Saxon Woods potential.

    Parameters
    ----------
    energy : Float
        DESCRIPTION. The energy of the scattering electron
    order : Int
        DESCRIPTION. The order of the bessel function and Saxon Woods potential

    Returns
    -------
    e_values : Float
        DESCRIPTION. Real part of phase shift

    '''
    
    rmin = 0.1
    rmax = 10.0

    rstep = (rmax - rmin) / 1000.0
    e_values = []
    e_values.append(0.0)
    
    energy_new = energy / 10.367    
    k_energy = sqrt(energy_new)
    
    for i in range(0, 1000):
        radius = rmin + i * rstep
        next_e = get_next_e(radius, k_energy, order, e_values[i], rstep)
        
        if isnan(next_e):    
            break
        else:
            e_values.append(next_e)
        
    return degrees(e_values[-1])


def plot_saxon_woods_real(order):
    '''
    Plot real phase shifts of Saxon Woods potential.

    Returns
    -------
    None.

    '''
    
    energy_grid = np.linspace(1, 50, num=25, endpoint=False)   
        
    e = []   
    for energy in energy_grid:
        phase_e = integrate_function(energy, order)
        e.append(phase_e)
   
    plt.xlabel('Ecm(MeV)')
    plt.ylabel('phase shift(degrees)')  
    plt.title('Partial wave l=4') 
    plt.plot(energy_grid, e, label = 'real; This Work')

    ax=plt.gca()   
    x_major_locator = MultipleLocator(3)
    y_major_locator=MultipleLocator(20)    
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    
    plt.legend()
    plt.show()

#Plot real phase shifts of Saxon Woods potential
print('Real phase shifts of Saxon Woods potential')
plot_saxon_woods_real(4)
