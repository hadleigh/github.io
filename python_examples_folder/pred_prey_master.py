"""
pred_prey_master.py

ENVR 300

H Thompson

March 2016
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pred_prey_integrator import Pred_Prey_Integrate

plt.style.use('ggplot')

test = Pred_Prey_Integrate('pred_prey.yaml')

rk_prey, rk_pred = test.runge_kutta_4()
euler_prey, euler_pred = test.euler_method()

t_vals = test.time_variables
tstart = t_vals.time_start
tend = t_vals.time_end
dt = t_vals.delta_t

time = np.arange(tstart, tend, dt)

plt.plot(time, euler_prey, label='Euler Method')
#plt.plot(time, euler_pred, label='Pred')
plt.plot(time, rk_prey, label='Runge Kutta Method')
#plt.plot(time, rk_pred, label='rk')
plt.title('Basic Predator-Prey Numerical Model Output \n 1/365 (one day) Time Step')
plt.xlabel('Time (yrs)')
plt.ylabel('Biomass * 100 ($kg. km^{2}$)')
plt.legend(loc='upper left')
plt.savefig('complex_model_compare7.png', format='png')
plt.show()
