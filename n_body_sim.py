import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. PHYSICS SETUP & CONSTANTS
# ==========================================
# Using Astronomical Units (AU), Solar Masses (M_sun), and Years
# In these units, the Gravitational constant G is exactly 4 * pi^2
G = 4 * np.pi**2

# Define the masses of our bodies (in Solar Masses)
# Body 0: The Star
# Body 1 & 2: Normal Planets
# Body 3: The "Missing Mass" (Planet X - dark, massive, distant)
masses = np.array([1.0, 3e-6, 4e-6, 0.005]) 
num_bodies = len(masses)

# ==========================================
# 2. THE DIFFERENTIAL EQUATIONS (The Engine)
# ==========================================
def n_body_equations(t, state):
    """
    Calculates the velocities and accelerations for all bodies at time t.
    State vector layout: [x1, y1, x2, y2... vx1, vy1, vx2, vy2...]
    """
    # Split the state vector into positions and velocities
    positions = state[:2 * num_bodies].reshape((num_bodies, 2))
    velocities = state[2 * num_bodies:].reshape((num_bodies, 2))
    
    # Accelerations start at zero
    accelerations = np.zeros_like(positions)
    
    # Calculate gravitational pull between every pair of bodies
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                # Vector from body i to body j
                r_vec = positions[j] - positions[i]
                # Distance squared and distance cubed (for the denominator)
                r_sq = np.sum(r_vec**2)
                r_cube = r_sq**1.5
                
                # Newton's Law of Universal Gravitation: a = G * m / r^2
                # Direction is handled by r_vec / r
                accelerations[i] += G * masses[j] * r_vec / r_cube
                
    # Flatten back into a 1D array for the SciPy solver
    d_state = np.concatenate([velocities.flatten(), accelerations.flatten()])
    return d_state

# ==========================================
# 3. INITIAL CONDITIONS
# ==========================================
# Format: [X, Y] in AU
initial_positions = np.array([
    [0.0, 0.0],      # Star (Center)
    [1.0, 0.0],      # Planet 1 (1 AU out)
    [1.5, 0.0],      # Planet 2 (1.5 AU out)
    [5.0, 0.0]       # Planet X (5 AU out - the hidden perturber)
])

# For a perfectly circular orbit, v = sqrt(G * M_star / r)
# Format: [Vx, Vy] in AU/Year
v1 = np.sqrt(G * masses[0] / 1.0)
v2 = np.sqrt(G * masses[0] / 1.5)
vX = np.sqrt(G * masses[0] / 5.0)

initial_velocities = np.array([
    [0.0, 0.0],      # Star is stationary
    [0.0, v1],       # Planet 1 moving "up"
    [0.0, v2],       # Planet 2 moving "up"
    [0.0, vX]        # Planet X moving "up"
])

# Combine into the initial state vector
initial_state = np.concatenate([initial_positions.flatten(), initial_velocities.flatten()])

# ==========================================
# 4. RUNNING THE SIMULATION
# ==========================================
# Simulate for 15 years, taking data points frequently
time_span = (0, 15)
time_eval = np.linspace(time_span[0], time_span[1], 2000)

print("Running N-Body Integration... (This involves solving complex differential equations)")
solution = solve_ivp(
    fun=n_body_equations,
    t_span=time_span,
    y0=initial_state,
    t_eval=time_eval,
    method='RK45', # Runge-Kutta 4th/5th order method
    rtol=1e-8,     # High precision tolerance
    atol=1e-8
)
print("Simulation Complete!")

# ==========================================
# 5. VISUALIZING THE DATA (The Output)
# ==========================================
# Extract position data for plotting
positions_over_time = solution.y[:2 * num_bodies].reshape((num_bodies, 2, len(time_eval)))

plt.figure(figsize=(10, 10))
plt.style.use('dark_background') # Looks more like space!

colors = ['yellow', 'cyan', 'green', 'magenta']
labels = ['Host Star', 'Planet A', 'Planet B', 'Unknown Mass (Planet X)']
alphas = [1.0, 0.8, 0.8, 0.3] # Make Planet X ghost-like

for i in range(num_bodies):
    x_data = positions_over_time[i, 0, :]
    y_data = positions_over_time[i, 1, :]
    
    # Plot the orbital trail
    plt.plot(x_data, y_data, color=colors[i], label=labels[i], alpha=alphas[i], linewidth=1.5)
    # Plot the final position as a dot
    plt.scatter(x_data[-1], y_data[-1], color=colors[i], s=100 if i==0 else 50, zorder=5)

plt.title("N-Body Gravitational Simulation: The Missing Mass Problem", fontsize=16, pad=20)
plt.xlabel("Distance (Astronomical Units)", fontsize=12)
plt.ylabel("Distance (Astronomical Units)", fontsize=12)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.grid(True, alpha=0.2)
plt.legend(loc='upper right', fontsize=10)
plt.axhline(0, color='grey', lw=0.5, alpha=0.5)
plt.axvline(0, color='grey', lw=0.5, alpha=0.5)
plt.tight_layout()
plt.show()
