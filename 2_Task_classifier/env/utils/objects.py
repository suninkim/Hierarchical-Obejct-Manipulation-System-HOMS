import math
import numpy as np
import pybullet_data
import pybullet as p

OBJECT_PATH = "./urdf/objects/"

def plane():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	plane_id = p.loadURDF('plane.urdf',
							basePosition=[0.0, 0.0, -0.554],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	return plane_id

def table():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	table_id = p.loadURDF(OBJECT_PATH+'table/table.urdf',
							basePosition=[0.584, 0.0, -0.026],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0,
							useFixedBase = True)
	return table_id

def wall1():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	wall1_id = p.loadURDF(OBJECT_PATH+'wall/wall.urdf',
							basePosition= [0.584 + 0.3 + 0.025 , 0.0, 1.04-0.554],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0,
							useFixedBase = True)
	return wall1_id

def wall2():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	wall2_id = p.loadURDF(OBJECT_PATH+'wall/wall.urdf',
							basePosition=[ 0.3+ 0.584, -0.625, 1.04-0.554],
							baseOrientation=[0.0, 0.0, 0.707107, 0.707107],
							globalScaling=1.0,
							useFixedBase = True)
	return wall2_id

def wall3():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	wall3_id = p.loadURDF(OBJECT_PATH+'wall/wall.urdf',
							basePosition=[ 0.3+ 0.584,  0.625, 1.04-0.554],
							baseOrientation=[0.0, 0.0, 0.707107, 0.707107],
							globalScaling=1.0,
							useFixedBase = True)
	return wall3_id

def box(useFixedBase=False):
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	box_id = p.loadURDF(OBJECT_PATH+'box/box.urdf',
							basePosition=[0.77, 0.35, -0.004],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=0.95,
							useFixedBase=useFixedBase)
	box_inner = [-1,0,2,4,6]
	box_out_list = [1,3,5,7]
	return box_id

def cover():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	cover_id = p.loadURDF(OBJECT_PATH+'cover/cover.urdf',
							basePosition=[0.77, 0.15,-0.004],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	cover_inner = [-1,1,3,5,7]
	cover_out_list = [0,2,4,6,8]
	return cover_id

def tray(useFixedBase=True):
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	tray_id = p.loadURDF(OBJECT_PATH+'tray_real/tray_real.urdf',
							basePosition=[0.45, -0.15, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0,
							useFixedBase=useFixedBase)
	return tray_id

def green():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	green_id = p.loadURDF(OBJECT_PATH+'green/green.urdf',
							basePosition=[0.48, -0.3, 0.0],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	return green_id

def blue():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	blue_id = p.loadURDF(OBJECT_PATH+'blue/blue.urdf',
							basePosition=[0.48, 0.4, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	return blue_id

def orange():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	orange_id = p.loadURDF(OBJECT_PATH+'orange/orange.urdf',
							basePosition=[0.55, 0.35, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	return orange_id

def laptop(useFixedBase=True):
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	laptop_id = p.loadURDF(OBJECT_PATH+'laptop/laptop.urdf',
							basePosition=[0.48, -0.2, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0,
							useFixedBase=useFixedBase)
	return laptop_id

def drawer(useFixedBase=True):
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	drawer_id = p.loadURDF(OBJECT_PATH+'drawer_1dan_real/drawer_1dan_real.urdf',
							basePosition=[0.60, 0.40, -0.009],
							baseOrientation=p.getQuaternionFromEuler([0, 0, 0.0]),
							globalScaling=1.0,
							useFixedBase=useFixedBase)
	return drawer_id

def shelf(useFixedBase=True):
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	shelf_id = p.loadURDF(OBJECT_PATH+'shelf/shelf.urdf',
							# front view
							# basePosition=[0.739, -0.393, -0.001],
							# baseOrientation=p.getQuaternionFromEuler([0, 0, -math.pi/2.0]),
							# side view
							basePosition=[0.678, -0.455, -0.004],
							baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi]),
							globalScaling=1.0,
							useFixedBase=useFixedBase)
	return shelf_id

def pencil_case(useFixedBase=True):
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	pencil_case_id = p.loadURDF(OBJECT_PATH+'pencil_case/pencil_case.urdf',
							basePosition=[0.77, 0.0, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0,
							useFixedBase=useFixedBase)
	return pencil_case_id

def board_marker():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	board_marker_id = p.loadURDF(OBJECT_PATH+'board_marker/board_marker.urdf',
							basePosition=[0.48, -0.25, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	return board_marker_id

def book():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	book_id = p.loadURDF(OBJECT_PATH+'book/book.urdf',
							basePosition=[0.77, -0.1, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	return book_id

def doll():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	doll_id = p.loadURDF(OBJECT_PATH+'doll/doll.urdf',
							basePosition=[0.4, -0.2, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	return doll_id

def lotion():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	lotion_id = p.loadURDF(OBJECT_PATH+'lotion/lotion.urdf',
							basePosition=[0.48, -0.2, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	return lotion_id

def tumbler():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	tumbler_id = p.loadURDF(OBJECT_PATH+'tumbler_cover/tumbler_cover.urdf',
							basePosition=[0.48, 0.4, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	return tumbler_id

def mouse():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	mouse_id = p.loadURDF(OBJECT_PATH+'mouse/mouse.urdf',
							basePosition=[0.5, -0.1, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	return mouse_id

def cube():
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	cube_id = p.loadURDF(OBJECT_PATH+'cube/cube.urdf',
							basePosition=[0.55, 0.35, -0.001],
							baseOrientation=[0, 0, 0, 1.0],
							globalScaling=1.0)
	return cube_id

def reset_object(body_id, position, orientation):
		p.resetBasePositionAndOrientation(body_id, position, orientation)

def get_object_position(body_id):
		object_position, object_orientation = \
				p.getBasePositionAndOrientation(body_id)
		return np.asarray(object_position), np.asarray(p.getEulerFromQuaternion(object_orientation))[2]

def get_object_link_info(body_id, link_index):
		position, orientation, _, _, _, _ = \
				p.getLinkState(body_id, link_index)
		return np.asarray(position), np.asarray(p.getEulerFromQuaternion(orientation))[2]

def get_object_joint_info(body_id, joint_index):
	state = p.getJointState(body_id, joint_index)
	joint_position = state[0]
	joint_velocity = state[1]

	return joint_position, joint_velocity

def cal_distance(goal_a, goal_b):
		assert goal_a.shape == goal_b.shape
		return np.linalg.norm(goal_a - goal_b, axis=-1)
