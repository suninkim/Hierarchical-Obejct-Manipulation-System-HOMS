import pybullet as p
import numpy as np

OBJECT_PATH = "./urdf/objects/"
XYZ_SCALE = 0.1
DEG_SCALE = 10.0

std = 0.075
ori_thresh = 0.25

def make_action(robotId, targetPoint, targetOrientation, dropPoint, dropOrientation, is_gripper_open, cur_point=0):
    
    eePos, eeOri, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(robotId, 11, computeLinkVelocity = 1)
    eePos, eeOri = np.asarray(eePos), np.asarray(p.getEulerFromQuaternion(eeOri))

    pos_1 = np.concatenate((targetPoint, (targetOrientation,)))
    pos_2 = pos_1.copy()
    pos_1[2] += 0.05
    pos_3 = pos_1.copy()
    pos_3[2] += 0.08
    pos_4 = np.concatenate((dropPoint, (dropOrientation,)))
    pos_5 = pos_4.copy()
    pos_4[2] += 0.08
    pos_6 = pos_4.copy()

    way_point = [pos_1,pos_2,pos_3,pos_4,pos_5,pos_6]
    action = np.zeros(6)
    noise = np.random.normal(0, std, 6)

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
        noise = np.random.normal(0, std, 6)
        noise[2] = 0
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point
    else:
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

def make_action_laptop_grasp(robotId, targetPoint, targetOrientation, dropPoint, dropOrientation, is_gripper_open, cur_point=0):
    
    eePos, eeOri, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(robotId, 11, computeLinkVelocity = 1)
    eePos, eeOri = np.asarray(eePos), np.asarray(p.getEulerFromQuaternion(eeOri))

    pos_0 = np.concatenate((targetPoint, (targetOrientation,)))
    pos_1 = pos_0.copy()
    pos_0[:3] = pos_0[:3] + np.array([0.02, -0.05, 0.24])
    pos_02 = pos_0.copy()
    pos_02[:3] = pos_02[:3] + np.array([-0.03, 0.0, 0.0])
    pos_2 = pos_1.copy()
    pos_1[2] += 0.05
    pos_3 = pos_1.copy()
    pos_1[0] -= 0.03
    pos_3[2] += 0.08
    pos_4 = np.concatenate((dropPoint, (dropOrientation,)))
    pos_5 = pos_4.copy()
    pos_4[2] += 0.08
    pos_6 = pos_4.copy()

    way_point = [pos_0, pos_02, pos_1,pos_2,pos_3,pos_4,pos_5,pos_6]
    action = np.zeros(6)
    noise = np.random.normal(0, std, 6)

    if cur_point < 8:
        target_pos =  way_point[cur_point]
    else:
        action[5] = 0.75
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

    if cur_point == 3 or cur_point == 6:
        grasp = False
    else:
        grasp = True


    action[3] = (1/DEG_SCALE)*np.clip(rad_to_deg(target_pos[3]) - rad_to_deg(eeOri[2]),-DEG_SCALE ,DEG_SCALE)
    action[:3] = (1/XYZ_SCALE)*np.clip(target_pos[:3] - eePos,-XYZ_SCALE, XYZ_SCALE)	

    if cur_point == 3 and is_gripper_open and np.linalg.norm(target_pos[:3] - eePos)<0.015:
        action[4] = -0.75
        action += noise
        cur_point += 1
        grasp = True
        return np.clip(action, -0.99, 0.99), cur_point

    if cur_point == 6 and not is_gripper_open:
        action[4] = 0.75
        action += noise
        cur_point += 1
        grasp = True
        return np.clip(action, -0.99, 0.99), cur_point

    if np.linalg.norm(target_pos[:3] - eePos) < 0.03 and grasp:
        action += noise
        cur_point += 1
        return np.clip(action, -0.99, 0.99), cur_point
    elif cur_point == 6:
        noise = np.random.normal(0, 0.1, 6)
        noise[2] = 0
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point
    else:
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

def make_action_laptop_close(robotId, prePoint, preOrientation, targetPoint, targetOrientation, is_gripper_open, cur_point=0):
        
    eePos, eeOri, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(robotId, 11, computeLinkVelocity = 1)
    eePos, eeOri = np.asarray(eePos), np.asarray(p.getEulerFromQuaternion(eeOri))

    ori_diff = abs(targetOrientation- eeOri[2])

    pos_1 = np.concatenate((prePoint, (preOrientation,)))
    pos_2 = pos_1.copy()
    pos_1[2] += 0.1
    pos_3 = pos_2.copy()    
    pos_3[:3] = pos_3[:3] + np.array([-0.07, 0.0, -0.03])    
    pos_4 = pos_3.copy()    
    pos_4[:3] = pos_4[:3] + np.array([-0.05, 0.0, -0.05])
    pos_5 = np.concatenate((targetPoint, (targetOrientation,)))

    way_point = [pos_1,pos_2,pos_3,pos_4,pos_5]
    action = np.zeros(6)
    noise = np.random.normal(0, std, 6)

    if cur_point < len(way_point):
        target_pos =  way_point[cur_point]
    else:
        action[5] = 0.75
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

    action[4] = -0.75
    grasp = True

    action[3] = (1/DEG_SCALE)*np.clip(rad_to_deg(target_pos[3]) - rad_to_deg(eeOri[2]),-DEG_SCALE ,DEG_SCALE)
    action[:3] = (1/XYZ_SCALE)*np.clip(target_pos[:3] - eePos,-XYZ_SCALE, XYZ_SCALE)	


    if np.linalg.norm(target_pos[:3] - eePos) < 0.035 and grasp and ori_diff < ori_thresh:
        action += noise
        cur_point += 1
        return np.clip(action, -0.99, 0.99), cur_point
    else:
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

def make_action_drawer_open(robotId, targetPoint, targetOrientation, targetVector, is_gripper_open, cur_point=0):
    
    eePos, eeOri, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(robotId, 11, computeLinkVelocity = 1)
    eePos, eeOri = np.asarray(eePos), np.asarray(p.getEulerFromQuaternion(eeOri))

    ori_diff = abs(targetOrientation- eeOri[2])

    pos_1 = np.concatenate((targetPoint, (targetOrientation,)))
    pos_2 = pos_1.copy()
    pos_1[2] += 0.05
    pos_3 = pos_2.copy()
    pos_3[:3] = pos_3[:3] - 0.18*targetVector
    pos_4 = pos_3.copy()
    pos_4[2] += 0.10

    way_point = [pos_1,pos_2,pos_3,pos_4]
    action = np.zeros(6)
    noise = np.random.normal(0, std, 6)

    if cur_point < len(way_point):
        target_pos =  way_point[cur_point]
    else:
        action[5] = 0.75
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

    if cur_point == 1 or cur_point ==2:
        grasp = False
    else:
        grasp = True

    action[3] = (1/DEG_SCALE)*np.clip(rad_to_deg(target_pos[3]) - rad_to_deg(eeOri[2]),-DEG_SCALE ,DEG_SCALE)
    action[:3] = (1/XYZ_SCALE)*np.clip(target_pos[:3] - eePos,-XYZ_SCALE, XYZ_SCALE)	

    if cur_point == 1 and is_gripper_open and np.linalg.norm(target_pos[:3] - eePos)<0.02 and ori_diff < ori_thresh:
        action[4] = -0.75
        action += noise
        cur_point += 1
        grasp = True
        return np.clip(action, -0.99, 0.99), cur_point

    if cur_point == 2 and not is_gripper_open and np.linalg.norm(target_pos[:3] - eePos) <0.02 and ori_diff < ori_thresh:
        action[4] = 0.75
        cur_point += 1
        grasp = True
        return np.clip(action, -0.99, 0.99), cur_point

    if np.linalg.norm(target_pos[:3] - eePos) < 0.025 and grasp and ori_diff < ori_thresh:
        action += noise
        cur_point += 1
        return np.clip(action, -0.99, 0.99), cur_point

    elif cur_point == 2:
        noise = np.random.normal(0, 0.05, 6)
        noise[2] = 0
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

    else:
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point
        
def make_action_drawer_close(robotId, targetPoint, targetOrientation, targetVector, is_gripper_open, cur_point=0):
    
    eePos, eeOri, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(robotId, 11, computeLinkVelocity = 1)
    eePos, eeOri = np.asarray(eePos), np.asarray(p.getEulerFromQuaternion(eeOri))

    ori_diff = abs(targetOrientation- eeOri[2])

    pos_1 = np.concatenate((targetPoint, (targetOrientation,)))	
    pos_1[:3] = pos_1[:3]
    pos_3 = pos_1.copy()
    pos_2 = pos_1.copy()
    pos_1[2] += 0.05
    pos_3[:3] = pos_3[:3] + 0.18*targetVector
    pos_4 = pos_3.copy()
    pos_4[2] += 0.10


    way_point = [pos_1,pos_2, pos_3, pos_4]
    action = np.zeros(6)
    noise = np.random.normal(0, std, 6) 

    if cur_point < len(way_point):
        target_pos =  way_point[cur_point]
    else:
        action[5] = 0.75
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

    if cur_point == 1 or cur_point ==2:
        grasp = False
    else:
        grasp = True


    action[3] = (1/DEG_SCALE)*np.clip(rad_to_deg(target_pos[3]) - rad_to_deg(eeOri[2]),-DEG_SCALE ,DEG_SCALE)
    action[:3] = (1/XYZ_SCALE)*np.clip(target_pos[:3] - eePos,-XYZ_SCALE, XYZ_SCALE)	

    if cur_point == 1 and is_gripper_open and np.linalg.norm(target_pos[:3] - eePos)<0.02 and ori_diff < ori_thresh:
        action[4] = -0.75
        action += noise
        cur_point += 1
        grasp = True
        return np.clip(action, -0.99, 0.99), cur_point

    if cur_point == 2 and not is_gripper_open and np.linalg.norm(target_pos[:3] - eePos) <0.025 and ori_diff < ori_thresh:
        action[4] = 0.75
        cur_point += 1
        grasp = True
        return np.clip(action, -0.99, 0.99), cur_point

    if np.linalg.norm(target_pos[:3] - eePos) < 0.025 and grasp and ori_diff < ori_thresh:
        action += noise
        cur_point += 1
        return np.clip(action, -0.99, 0.99), cur_point

    elif cur_point == 2:
        noise = np.random.normal(0, 0.05, 6)
        noise[2] = 0
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

    else:
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

def make_action_box_push(robotId, targetPoint, targetOrientation, targetVector, is_gripper_open, cur_point=0):
        
    eePos, eeOri, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(robotId, 11, computeLinkVelocity = 1)
    eePos, eeOri = np.asarray(eePos), np.asarray(p.getEulerFromQuaternion(eeOri))

    ori_diff = abs(targetOrientation- eeOri[2])

    pos_1 = np.concatenate((targetPoint, (targetOrientation,)))
    pos_2 = pos_1.copy()
    pos_1[2] += 0.05
    pos_3 = pos_2.copy()
    pos_3[:3] = pos_3[:3] + 0.30*targetVector
    pos_4 = pos_3.copy()
    pos_4[2] += 0.10

    way_point = [pos_1,pos_2,pos_3,pos_4]
    action = np.zeros(6)
    noise = np.random.normal(0, std, 6)

    if cur_point < len(way_point):
        target_pos =  way_point[cur_point]
    else:
        action[5] = 0.75
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

    action[4] = -0.75
    grasp = True

    action[3] = (1/DEG_SCALE)*np.clip(rad_to_deg(target_pos[3]) - rad_to_deg(eeOri[2]),-DEG_SCALE ,DEG_SCALE)
    action[:3] = (1/XYZ_SCALE)*np.clip(target_pos[:3] - eePos,-XYZ_SCALE, XYZ_SCALE)
    
    if cur_point in [2,3] and np.linalg.norm(target_pos[:3] - eePos) < 0.035 and grasp and ori_diff < ori_thresh:
        action += noise
        cur_point += 1
        return np.clip(action, -0.99, 0.99), cur_point

    if np.linalg.norm(target_pos[:3] - eePos) < 0.03 and grasp and ori_diff < ori_thresh:
        action += noise
        cur_point += 1
        return np.clip(action, -0.99, 0.99), cur_point
    else:
        action += noise
        return np.clip(action, -0.99, 0.99), cur_point

def rad_to_deg(rad):
    return rad * 180. / np.pi
