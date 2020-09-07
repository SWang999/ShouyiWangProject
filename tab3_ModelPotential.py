from math import cos
from math import sin
from math import exp
from math import cosh
from math import sinh

from scipy.special import spherical_jn
from scipy.special import spherical_yn

#In order to calculate Table 3, please use the file of tab3_SciPySaxonWoods
#This file is only used to calculate Saxon Woods potential
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


class OpticalPotential:
      
    def update_scattering_energy(self, energy):
        self.energy = energy

    def update_order(self, order):
        self.order = order
    
    def get_optical_potential(self, radius):
        '''
        According formulas to calculate the value of Saxon Woods potential.

        Parameters
        ----------
        radius : Float
            DESCRIPTION. The radius away from the target

        Returns
        -------
        TYPE: Complex
            DESCRIPTION. Complex value of Saxon Woods potential

        '''

        if self.order == 0:
            potreal = (150.0 / (1.0 + exp((radius - 1.65) / 0.10))) \
                    - (9.2 / (1.0 + exp((radius - 3.72) / 0.40)))
            potimag = - 5.0 / (1.0 + exp((radius - 3.72)/ 0.40))
        elif self.order == 2:        
            potreal = (150.0 / (1.0 + exp((radius - 1.63) / 0.05))) \
                        - (16.0 / (1.0 + exp((radius - 3.55) / 0.3)))
            potimag = - 5.0 / (1.0 + exp((radius - 3.55)/ 0.3))
        elif self.order == 4:
            potreal = (220.0 / (1.0 + exp((radius - 1.20) / 0.05))) \
                        - (71.0 / (1.0 + exp((radius - 2.48) / 0.46)))
            potimag = - 5.0 / (1.0 + exp((radius - 2.48)/ 0.46)) 
        else:
            print('Order number error, please enter the order number equal to 0, 2 or 4')
        
        #unit conversion
        real = potreal / 10.367
        imag = potimag / 10.367   
        
        return complex(real, imag) 

    def get_phases_coupled(self, radius, phases):
        '''
        Given the previous values of the phases, calculate the new phases.

        Parameters
        ----------
        radius : Float
            DESCRIPTION. The radius away from the target
        phases : Float
            DESCRIPTION. The previous real and imag parts of phase 

        Returns
        -------
        list
            DESCRIPTION. The new real and imag parts of phase

        '''
        
        #Jain's formula
        rb_first = riccati_bessel_first(self.order, radius * self.energy)
        rb_second = riccati_bessel_second(self.order, radius * self.energy)

        X = cosh(phases[1]) * (rb_second * sin(phases[0]) - rb_first * cos(phases[0]))
        Y = sinh(phases[1]) * (rb_second * cos(phases[0]) + rb_first * sin(phases[0]))
        potential_value = self.get_optical_potential(radius)

        dy_0 = -(1.0/self.energy) * (potential_value.real * (X ** 2 - Y ** 2) - 2 * potential_value.imag * X * Y)
        dy_1 = -(1.0/self.energy) * (2 * potential_value.real * X * Y + potential_value.imag * (X ** 2 - Y ** 2))

        return [dy_0, dy_1]
    
    def get_phases_single(self, radius, phases):
        '''
        Given the previous values of the phases, calculate the new phases.

        Parameters
        ----------
        radius : Float
            DESCRIPTION. The radius away from the target
        phases : Float
            DESCRIPTION. The previous real and imag parts of phase 

        Returns
        -------
        list
            DESCRIPTION. The new real and imag parts of phase

        '''
        
        #Jana's formula
        rb_first = riccati_bessel_first(self.order, radius * self.energy)
        rb_second = riccati_bessel_second(self.order, radius * self.energy)

        prefactor = - 1 /  self.energy
        
        cos_phase = complex(cos(phases[0]) * cosh(phases[1]), - sin(phases[0]) * sinh(phases[1])) 
        sin_phase = complex(sin(phases[0]) * cosh(phases[1]), cos(phases[0]) * sinh(phases[1]))        
        sl_e = cos_phase * rb_first - sin_phase * rb_second    
        
        potential_value = self.get_optical_potential(radius)
        new_phase_e = prefactor * potential_value * sl_e * sl_e
        
        return [new_phase_e.real, new_phase_e.imag]
