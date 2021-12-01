import numpy as np
import pybullet as p

def connect_headless(gui=False):
	if gui:
		cid = p.connect(p.SHARED_MEMORY)
		if cid < 0:
			p.connect(p.GUI)
	else:
		p.connect(p.DIRECT)

	p.setRealTimeSimulation(False)
	p.stepSimulation()

def disconnect():
	p.disconnect()

def setup(timestep=1./240, solver_iterations=150, gravity=-9.8):
	p.setPhysicsEngineParameter(numSolverIterations=solver_iterations)
	p.setTimeStep(timestep)
	p.setGravity(0, 0, gravity)
	p.stepSimulation()

def step_simulation(num_sim_steps):
    for _ in range(num_sim_steps):
        p.stepSimulation()