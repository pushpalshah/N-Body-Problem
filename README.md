# N-Body-Problem
# N-Body Gravitational Integrator: "The Missing Mass"

## Overview
This repository bridges rigorous computational astrophysics with STEM outreach. It contains a custom Newtonian physics engine built from scratch to solve the differential equations of motion for multi-planetary systems. 

Furthermore, this simulation is designed as an interactive educational module. By introducing an invisible, massive object into a stable system, students are challenged to use gravitational perturbations to locate the "Missing Mass" (representing dark matter or an undiscovered planetary body).

## Computational Physics 
Unlike pre-built physics engines, this script manually defines Newton's Law of Universal Gravitation and utilizes high-end numerical integrators to solve the initial value problems (IVPs).

**Skills Demonstrated:**
* Custom Physics Engine Development
* Vector Mathematics & Differential Equations
* 4th-Order Runge-Kutta Integration (`scipy.integrate.solve_ivp`)
* Educational System Modeling

## Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the simulation: `python n_body_sim.py`

*Note: For the outreach exercise, the final plotting code can be modified to hide "Planet X," challenging students to calculate its location based purely on the orbital warping of the visible planets.*
