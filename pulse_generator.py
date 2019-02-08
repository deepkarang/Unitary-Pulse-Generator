#Import external dependencies
import numpy as np
import matplotlib.pyplot as plt
import time

#Import QuTiP basic operating modules
from qutip import Qobj, identity, sigmax, sigmaz
from qutip.qip import hadamard_transform
import qutip.logging_utils as logging
logger = logging.get_logger()
log_level = logging.INFO

#Import QuTiP control modules
import qutip.control.pulseoptim as cpo

#Define target transformation name
example_name = 'Hadamard'

# Defining components for initial and target states
Drift_Hamiltonian = sigmaz()
Control_hamiltonian = [sigmax()]
Initial_unitary = identity(2)
Target_unitary = hadamard_transform(1)

# Define time evolution parameters
num_timesteps = 10
evolution_time = 10

# Pulse optimization termination conditions
fidelity_error_required = 1e-10
max_iter = 200
max_wall_time = 120

# Minimum gradient. As this tends to 0 -> local minima has been found
minimum_gradient = 1e-20

# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'SINE'

# Start timer
start_time = time.time()

# Run optimization
result = cpo.optimize_pulse_unitary(Drift_Hamiltonian, Control_hamiltonian, Initial_unitary, Target_unitary, num_timesteps, evolution_time, 
                fid_err_targ=fidelity_error_required, min_grad=minimum_gradient, 
                max_iter=max_iter, max_wall_time=max_wall_time, init_pulse_type=p_type, 
                log_level=log_level, gen_stats=True)

# Output results
result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print(result.termination_reason)
print("Optimization duration: " + str(time.time() - start_time) + " seconds")

plot = plt.figure()
axis1 = plot.add_subplot(2, 1, 1)
axis1.set_title("Initial control amplitudes")
axis1.set_ylabel("Control amplitude")
axis1.step(result.time,
         np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])),
         where='post')

axis2 = plot.add_subplot(2, 1, 2)
axis2.set_title("Optimised Control Sequences")
axis2.set_xlabel("Time")
axis2.set_ylabel("Control amplitude")
axis2.step(result.time,
         np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])),
         where='post')
plt.tight_layout()
plt.show()
