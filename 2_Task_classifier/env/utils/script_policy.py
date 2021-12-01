import pybullet_data
import pybullet as p
import os
import numpy as np

OBJECT_PATH = "./urdf/objects/"
XYZ_SCALE = 0.2
DEG_SCALE = 20.0

box_low = [0.45, 0.3, -0.008, -0.15]
cover_low = [0.65, -0.05, -0.008, -0.2]
object_low = [0.42, -0.05, 0.015, -0.3]
obstacle_low = []
box_high = [0.52, 0.4, -0.008,  0.15]
cover_high = [0.7, 0.1, -0.008, 0.2]
object_high = [0.5, 0.16, 0.015, 0.3]
obstacle_high = []

def rad_to_deg(rad):
    # return np.array([ for r in rad])
    return rad * 180. / np.pi

def grasping_action(targetPoint, targetOrientation, objecHeight, is_gripper_open, threshold=0.02, height = 0.15):
	action = np.zeros((6,))
	eePos, eeOri, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(9, 11, computeLinkVelocity = 1)
	eeOri = p.getEulerFromQuaternion(eeOri)
	
	targetEEDist = np.linalg.norm(np.asarray(targetPoint)-np.asarray(eePos))
	if targetEEDist > threshold:
		action[:3] = np.asarray(targetPoint) - np.asarray(eePos)
		action[3] = targetOrientation - eeOri[2]
	elif is_gripper_open:
		action[4] = -0.75
	elif objecHeight < height:
		action[2] = 1.0
	else:
		action = np.zeros((6))


	noise = np.random.normal(0, 0.1, 6)
	action = np.clip(10*action,-0.99, 0.99)# + noise

	return action

def make_action(robotId, targetPoint, targetOrientation, dropPoint, dropOrientation, is_gripper_open, cur_point=0):
	
	eePos, eeOri, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(robotId, 11, computeLinkVelocity = 1)
	eePos, eeOri = np.asarray(eePos), np.asarray(p.getEulerFromQuaternion(eeOri))

	pos_1 = np.concatenate((targetPoint, (targetOrientation[2],)))
	pos_2 = pos_1.copy()
	# pos_2[2] += 0.04
	pos_1[2] += 0.06
	pos_3 = pos_1.copy()
	pos_3[2] += 0.025	
	pos_4 = np.concatenate((dropPoint, (dropOrientation,)))
	pos_5 = pos_4.copy()
	pos_4[2] += 0.02
	pos_6 = pos_4.copy()
	# pos_6[2] += 0.05

	way_point = [pos_1,pos_2,pos_3,pos_4,pos_5,pos_6]
	action = np.zeros(6)
	noise = np.random.normal(0, 0.1, 6)

	if cur_point < 6:
		target_pos =  way_point[cur_point]
	else:
		action[5] = 0.75
		action += noise
		return np.clip(action, -0.99, 0.99), cur_point

	if cur_point == 1 or cur_point == 4:
		grasp = False
	else:
		grasp = True


	action[3] = (1/DEG_SCALE)*np.clip(rad_to_deg(target_pos[3]) - rad_to_deg(eeOri[2]),-DEG_SCALE ,DEG_SCALE)
	# print(action[3])
	action[:3] = (1/XYZ_SCALE)*np.clip(target_pos[:3] - eePos,-XYZ_SCALE, XYZ_SCALE)	

	if cur_point == 1 and is_gripper_open and np.linalg.norm(target_pos[:3] - eePos)<0.015:
		action[4] = -0.75
		action += noise
		cur_point += 1
		grasp = True
		return np.clip(action, -0.99, 0.99), cur_point

	if cur_point == 4 and not is_gripper_open:
		action[4] = 0.75
		action += noise
		cur_point += 1
		grasp = True
		return np.clip(action, -0.99, 0.99), cur_point

	if np.linalg.norm(target_pos[:3] - eePos) < 0.025 and grasp:
		action += noise
		cur_point += 1
		return np.clip(action, -0.99, 0.99), cur_point
	elif cur_point == 2:
		noise = np.random.normal(0, 0.1, 6)
		noise[2] = 0
		action += noise
		return np.clip(action, -0.99, 0.99), cur_point

	else:
		action += noise
		return np.clip(action, -0.99, 0.99), cur_point
		

def make_action_stage4(robotId, targetPoint, targetOrientation, is_gripper_open, cur_point=0):
	
	eePos, eeOri, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(robotId, 11, computeLinkVelocity = 1)
	eePos, eeOri = np.asarray(eePos), np.asarray(p.getEulerFromQuaternion(eeOri))

	pos_1 = np.concatenate((targetPoint, (targetOrientation[2],)))
	pos_2 = pos_1.copy()
	pos_2[1] = 0.34


	way_point = [pos_1,pos_2]
	action = np.zeros(6)
	noise = np.random.normal(0, 0.1, 6)

	if cur_point < 2:
		target_pos =  way_point[cur_point]
	else:
		action[5] = 0.75
		action += noise
		return np.clip(action, -0.99, 0.99), cur_point

	grasp = True

	action[3] = (1/DEG_SCALE)*np.clip(rad_to_deg(target_pos[3]) - rad_to_deg(eeOri[2]),-DEG_SCALE ,DEG_SCALE)
	action[:3] = (1/XYZ_SCALE)*np.clip(target_pos[:3] - eePos,-XYZ_SCALE, XYZ_SCALE)	

	if np.linalg.norm(target_pos[:3] - eePos) < 0.025 and grasp:
		action[4] = -0.75
		action += noise
		cur_point += 1
		return np.clip(action, -1.0, 1.0), cur_point

	else:
		action[4] = -0.75
		action += noise
		return np.clip(action, -0.99, 0.99), cur_point

def make_action_stage5(robotId, targetPoint, targetOrientation, is_gripper_open, cur_point=0):
	
	eePos, eeOri, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(robotId, 11, computeLinkVelocity = 1)
	eePos, eeOri = np.asarray(eePos), np.asarray(p.getEulerFromQuaternion(eeOri))

	pos_1 = np.concatenate((targetPoint, (targetOrientation[2],)))
	pos_1[1] -= 0.05
	pos_2 = np.concatenate((targetPoint, (targetOrientation[2],)))
	pos_3 = pos_2.copy()
	pos_3[0] = 0.60


	way_point = [pos_1,pos_2,pos_3]
	action = np.zeros(6)
	noise = np.random.normal(0, 0.1, 6)

	if cur_point < 3:
		target_pos =  way_point[cur_point]
	else:
		action[5] = 0.75
		action += noise
		return np.clip(action, -0.99, 0.99), cur_point

	grasp = True

	action[3] = (1/DEG_SCALE)*np.clip(rad_to_deg(target_pos[3]) - rad_to_deg(eeOri[2]),-DEG_SCALE ,DEG_SCALE)
	action[:3] = (1/XYZ_SCALE)*np.clip(target_pos[:3] - eePos,-XYZ_SCALE, XYZ_SCALE)	

	if np.linalg.norm(target_pos[:3] - eePos) < 0.025 and grasp:
		action[4] = -0.75
		action += noise
		cur_point += 1
		return np.clip(action, -0.99, 0.99), cur_point

	else:
		action[4] = -0.75
		action += noise
		return np.clip(action, -0.99, 0.99), cur_point


def make_action_stage6(robotId, str_pos, str_ori, end_pos, end_ori, is_gripper_open, cur_point=0):
	
	eePos, eeOri, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(robotId, 11, computeLinkVelocity = 1)
	eePos, eeOri = np.asarray(eePos), np.asarray(p.getEulerFromQuaternion(eeOri))

	pos_1 = np.concatenate((str_pos, (str_ori[2],)))
	pos_1[2] += 0.07
	pos_1[0] += 0.05
	pos_2 = np.concatenate((str_pos, (str_ori[2],)))
	pos_3  =np.concatenate((end_pos, (end_ori[2],)))

	# print(cur_point)


	way_point = [pos_1,pos_2, pos_3]
	action = np.zeros(6)
	noise = np.random.normal(0, 0.1, 6)

	if cur_point < 3:
		target_pos =  way_point[cur_point]
	else:
		action[5] = 0.75
		action += noise
		return np.clip(action, -0.99, 0.99), cur_point

	grasp = True

	action[3] = (1/DEG_SCALE)*np.clip(rad_to_deg(target_pos[3]) - rad_to_deg(eeOri[2]),-DEG_SCALE ,DEG_SCALE)
	action[:3] = (1/XYZ_SCALE)*np.clip(target_pos[:3] - eePos,-XYZ_SCALE, XYZ_SCALE)	

	if np.linalg.norm(target_pos[:3] - eePos) < 0.025 and grasp:
		action[4] = -0.75
		action += noise
		cur_point += 1
		return np.clip(action, -0.99, 0.99), cur_point

	else:
		action[4] = -0.75
		action += noise
		return np.clip(action, -0.99, 0.99), cur_point