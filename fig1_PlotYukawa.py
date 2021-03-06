import matplotlib.pyplot as plt

from scipy.special import spherical_jn
from scipy.special import spherical_yn
from math import sin
from math import exp
from math import atan
from math import isnan


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


def get_real_potential(radius):
    '''
    According formulas to calculate the value of Yukawa potential.

    Parameters
    ----------
    radius : Float
        DESCRIPTION. The radius away from the target

    Returns
    -------
    potreal : Float
        DESCRIPTION. Real potential value of Yukawa potential

    '''  
    
    potreal = (20/radius) * exp(-2*radius) - (10/radius) * exp(-radius)

    return potreal


def get_phases(potential_value, radius, energy, order, previous_phase_e):
    '''
    Given the previous values of the phases, calculate the new phases.

    Parameters
    ----------
    potential_value : Float
        DESCRIPTION. Real potential value of Yukawa potential
    radius : Float
        DESCRIPTION. The radius away from the target
    energy : Float
        DESCRIPTION. The energy of the scattering electron (eV)
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
    d_l = rb_first * rb_first + rb_second * rb_second
    s_l = - atan(rb_first/ rb_second)
    prefactor = - d_l /  energy
    sin_sl_e = sin(s_l + previous_phase_e)
    
    new_phase_e = prefactor * potential_value * sin_sl_e * sin_sl_e
    
    return new_phase_e


def get_next_e(radius, energy, order, e_value, step):
    '''
    The formula of Runge-Kutta method, which is used to calculate next phase shift value.

    Parameters
    ----------
    radius : Float
        DESCRIPTION. The radius away from the target
    energy : Float
        DESCRIPTION. The energy of the scattering electron (eV)
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
        potential_value.append(get_real_potential(rvalue))
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
    Integration over the Yukawa potential.

    Parameters
    ----------
    energy : Float
        DESCRIPTION. The energy of the scattering electron (eV)
    order : Int
        DESCRIPTION. The order of the bessel function

    Returns
    -------
    e_values : Float
        DESCRIPTION. Real part of phase shift

    '''
    
    rmin = 0.01
    rmax = 6.0
    rstep = (rmax - rmin) / 1000.0
    e_values = []
    e_values.append(0.0)

    for i in range(0, 1000):
        radius = rmin + i * rstep
        next_e = get_next_e(radius, energy, order, e_values[i],
                            rstep)
        if isnan(next_e):    
            break
        else:
            e_values.append(next_e)
    
    return e_values


def plot_real_potential():
    '''
    Plot phase shifts of Yukawa potential.

    Returns
    -------
    None.

    '''
    
    rmin = 0.01
    rmax = 6.0
    rstep = (rmax - rmin) / 1000.0
    r_values = []
    r_values.append(0.0)
    for i in range(0, 1000):
        radius = rmin + i * rstep
        r_values.append(radius)    
       
    plt.xlabel('r')
    plt.ylabel('phase shift')
    
    #the unit of k is eV
    k = [1, 10]
    
    e_0 = integrate_function(k[0], 0)    
    plt.plot(r_values, e_0, label = 'this work:k = 1, l=0', color='b')
    e_1 = integrate_function(k[0], 1)    
    plt.plot(r_values, e_1, label = 'k = 1, l=1', color='r')  
    e_2 = integrate_function(k[0], 2)    
    plt.plot(r_values, e_2, label = 'k = 1, l=2', color='m')    
    e_3 = integrate_function(k[1], 0)    
    plt.plot(r_values, e_3, label = 'k = 10, l=0', color='g')
    e_4 = integrate_function(k[1], 1)    
    plt.plot(r_values, e_4, label = 'k = 10, l=1', color='c')
    e_5 = integrate_function(k[1], 2)    
    plt.plot(r_values, e_5, label = 'k = 10, l=2', color='y')

    plt.legend()
    plt.show()

#Plot phase shifts of Yukawa potential
print('Phase shifts of Yukawa potential')
plot_real_potential()

