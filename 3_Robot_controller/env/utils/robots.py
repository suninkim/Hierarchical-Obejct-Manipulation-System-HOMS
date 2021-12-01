from collections import namedtuple
from attrdict import AttrDict
import pybullet_data
import pybullet as p
import os
import numpy as np

def Panda():
    robot_id = p.loadURDF("./urdf/franka_panda/panda.urdf",
                          basePosition=[0.0, 0.0, 0.0],
                          baseOrientation=[0, 0, 0, 1.0],
                          globalScaling=1.0,
                          useFixedBase = True)
    return robot_id

def setup_joint_control(robotID):
    controlJoints = ["panda_joint1","panda_joint2",
                     "panda_joint3", "panda_joint4",
                     "panda_joint5", "panda_joint6", "panda_joint7",
                     "panda_finger_joint1",
                     "panda_finger_joint2"]
    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    # print(numJoints)
    jointInfo = namedtuple("jointInfo", 
                           ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
    joints = AttrDict()
    joint_indices = []
    JOINT_LIMIT_LOWER = []
    JOINT_LIMIT_UPPER = []
    JOINT_RANGE = []
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID,jointName,jointType,jointLowerLimit,
                         jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
        if info.type=="REVOLUTE": # set revolute joint to static
            p.setJointMotorControl2(robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        if info.type=="PRISMATIC": # set revolute joint to static
            p.setJointMotorControl2(robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
        if controllable:
            joint_indices.append(i)
            JOINT_LIMIT_LOWER.append(jointLowerLimit)
            JOINT_LIMIT_UPPER.append(jointUpperLimit)
            JOINT_RANGE.append(jointUpperLimit-jointLowerLimit)
 
    return JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER, JOINT_RANGE, joint_indices

def reset_robot(robot_id, reset_joint_indices, reset_joint_values):
    assert len(reset_joint_indices) == len(reset_joint_values)
    for i, value in zip(reset_joint_indices, reset_joint_values):
        p.resetJointState(robot_id, i, value)
        p.setJointMotorControl2(robot_id,
                            i,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=value,
                            targetVelocity=0,
                            force=500,
                            positionGain=0.05)

def move_to_neutral(robot_id, reset_joint_indices, reset_joint_values,
                    num_sim_steps=75):
    assert len(reset_joint_indices) == len(reset_joint_values)
    p.setJointMotorControlArray(robot_id,
                                reset_joint_indices,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=reset_joint_values,
                                forces=[100] * len(reset_joint_indices),
                                positionGains=[0.03] * len(reset_joint_indices),
                                )
    for _ in range(num_sim_steps):
        p.stepSimulation()

def get_link_state(body_id, link_index):
    position, orientation, _, _, _, _, xyz_vel, abc_vel = p.getLinkState(body_id, link_index, computeLinkVelocity = 1)
    return np.asarray(position), np.asarray(p.getEulerFromQuaternion(orientation)), np.asarray(xyz_vel), np.asarray(abc_vel)

def get_joint_states(body_id, joint_indices):
    all_joint_states = p.getJointStates(body_id, joint_indices)
    joint_positions, joint_velocities = [], []
    for state in all_joint_states:
        joint_positions.append(state[0])
        joint_velocities.append(state[1])

    return np.asarray(joint_positions), np.asarray(joint_velocities)


def get_gripper_state(body_id, joint_indices):
    all_joint_states = p.getJointStates(body_id, joint_indices[-2:])
    joint_positions, joint_velocities = [], []
    for state in all_joint_states:
        joint_positions.append(state[0])
        joint_velocities.append(state[1])

    return np.asarray(joint_positions), np.asarray(joint_velocities), np.asarray(joint_positions).sum()

def apply_action_ik(target_ee_pos, target_ee_quat, target_gripper_state,
                    robot_id, end_effector_index, movable_joints,
                    lower_limit, upper_limit, rest_pose, joint_range,
                    num_sim_steps=5):
    # print(lower_limit)
    # print(upper_limit)
    # print(joint_range)
    # print(rest_pose)
    joint_poses = p.calculateInverseKinematics(robot_id,
                                               end_effector_index,
                                               target_ee_pos,
                                               target_ee_quat,
                                               # lowerLimits=lower_limit,
                                               # upperLimits=upper_limit,
                                               # jointRanges=joint_range,
                                               restPoses=rest_pose,
                                               jointDamping=[0.001] * len(
                                                   movable_joints),
                                               solver=0,
                                               maxNumIterations=100,
                                               residualThreshold=.005)

    p.setJointMotorControlArray(robot_id,
                                movable_joints,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=joint_poses,
                                # targetVelocity=0,
                                forces=[100] * len(movable_joints),
                                positionGains=[0.03] * len(movable_joints),
                                # velocityGain=1
                                )
    # set gripper action
    p.setJointMotorControl2(robot_id,
                            movable_joints[-2],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[0],
                            force=500,
                            positionGain=0.05)
    p.setJointMotorControl2(robot_id,
                            movable_joints[-1],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[1],
                            force=500,
                            positionGain=0.05)

    for _ in range(num_sim_steps):
        p.stepSimulation()

def quat_to_deg(quat):
    euler_rad = p.getEulerFromQuaternion(quat)
    euler_deg = rad_to_deg(euler_rad)
    return euler_deg


def deg_to_quat(deg):
    rad = deg_to_rad(deg)
    quat = p.getQuaternionFromEuler(rad)
    return quat


def deg_to_rad(deg):
    return np.array([d * np.pi / 180. for d in deg])


def rad_to_deg(rad):
    return np.array([r * 180. / np.pi for r in rad])
