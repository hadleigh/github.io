"""
pred_prey_integrator.py

ENVR 300

H Thompson

March 2016
"""
import numpy as np 
import yaml
from collections import namedtuple

class Pred_Prey_Integrate:
	
	def __init__(self, coeff_file):
		with open(coeff_file, 'rb') as f:
			config = yaml.load(f)
		self.config = config

		time_variables = namedtuple('time_variables', config['time_variables'].keys())
		initial_values = namedtuple('initial_values', config['initial_values'].keys())
		user_values = namedtuple('user_values', config['user_values'].keys())

		self.time_variables = time_variables(**config['time_variables'])
		self.initial_values = initial_values(**config['initial_values'])
		self.user_values = user_values(**config['user_values'])


	def __str__(self):
		out = 'Runge-Kutta Integrator with attributes time_variables, initial_values, ' + \
			'user_values. NOTE: In all function: V = Prey, P = Predator'
		return out

	def derivs(self, V, P):
		"""
		sefl, float x 2 --> float x 2

		supplies next time-step values of the derivative given current time step values
		of V and P

		returns v_next_step, p_next_step
		"""
		# coeff from yaml file
		vals = self.user_values
		a = vals.a
		b = vals.b
		c = vals.c
		d = vals.d
		K = vals.K

		# Prey equation
		v_next_step = a*V - (b*V*P)

		# Predator equation
		p_next_step = (a*c*V*P) - (d*P)

		vals = np.empty([2,1])
		vals[0] = v_next_step
		vals[1] = p_next_step

		return vals


	def runge_kutta_4(self):
		"""
		self --> np.ndarray x 2

		uses 4th order runge kutta to solve predator prey equations exclusively

		"""
		# time values from yaml file
		t = self.time_variables
		tstart = t.time_start
		tend = t.time_end
		dt = t.delta_t

		# initial values from yaml file
		i = self.initial_values
		p_old = i.pred_initial
		v_old = i.prey_initial

		time_steps = np.arange(tstart, tend, dt)
		n = len(time_steps)
		h = dt

		# creating V and P lists to stick the results
		p_vals = []
		p_vals.append(p_old)

		v_vals = []
		v_vals.append(v_old)

		# loop dependent on length of time steps array
		for i in range(n-1):

			# first intermediate step using initial values
			m1_k1 = h * (self.derivs(v_old,p_old))
			m1 = m1_k1[0]
			k1 = m1_k1[1]

			# second intermediate step
			m2_k2 = h * (self.derivs((v_old+(m1/2.)), (p_old+(k1/2.))))
			m2 = m2_k2[0]
			k2 = m2_k2[1]

			# third intermediate step
			m3_k3 = h * (self.derivs((v_old+(m2/2.)), (p_old+(k2/2.))))
			m3 = m3_k3[0]
			k3 = m3_k3[1]

			# forth intermediate step
			m4_k4 = h * (self.derivs((v_old+m3), (p_old+k3)))
			m4 = m4_k4[0]
			k4 = m4_k4[1]

			# calc the new value and append it to the list of values
			v_new = v_old + ((m1 + (2.*m2) + (2.*m3) + (m4))/6.)
			v_vals.append(v_new)
			
			p_new = p_old + ((k1 + (2.*k2) + (2.*k3) + (k4))/6.)
			p_vals.append(p_new)
			
			# new values become new initial values (v-old, p_old)
			v_old = v_new
			p_old = p_new
			 
		p_vals = np.array(p_vals).squeeze()
		v_vals = np.array(v_vals).squeeze()

		return v_vals, p_vals


	def euler_method(self):
		"""
		self --> np.ndarray x 2

		uses simple euler method for calculating the derivative

		returns p_vals, v_vals
		"""

		# time valuess from yaml file
		t = self.time_variables
		tstart = t.time_start
		tend = t.time_end
		dt = t.delta_t
		time_steps = np.arange(tstart, tend, dt)
		n = len(time_steps)

		# coeff from yaml file
		vals = self.user_values
		a = vals.a
		b = vals.b
		c = vals.c
		d = vals.d

		# initial values from yaml file
		iv = self.initial_values
		v_old = iv.prey_initial
		p_old = iv.pred_initial

		# intilizing the lists to hold the values
		v_vals = []
		p_vals = []
		v_vals.append(v_old)
		p_vals.append(p_old)

		for i in range(n-1):

			# next time step calc of derivative
			v_new = v_old + (a*v_old*dt) - (b*v_old*p_old*dt)
			p_new = p_old + (c*v_old*p_old*dt) - (d*p_old*dt)

			# append the lists with the new values
			v_vals.append(v_new)
			p_vals.append(p_new)

			# new values then become the initial values for the next loop
			v_old = v_new
			p_old = p_new

		v_vals = np.array(v_vals).squeeze()
		p_vals = np.array(p_vals).squeeze()

		return v_vals, p_vals












		