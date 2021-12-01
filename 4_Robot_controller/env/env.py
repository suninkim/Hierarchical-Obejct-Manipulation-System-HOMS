import os
import gym
import time
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from collections import deque

from .utils import objects, robots, simulator, task_policy

END_EFFECTOR_INDEX = 11
RESET_JOINT_VALUES = [0.3, 0.2, 0.0, -2.0, 0.0, 2.2,  0.7963762844+0.3, 0.04, 0.04]
RESET_JOINT_VALUES_GRIPPER_CLOSED = [0.3, 0.2, 0.0, -2.0, 0.0, 2.2,  0.7963762844+0.3, 0., -0.]
ACTION_RANGE = 1.0

XYZ_SCALE = 0.1 # meter
DEG_SCALE = 10.0 # degree

MIN_DEP = 0.01
MAX_DEP = 1.1
SEG_NUM = 9
img_size = 256

class PandaEnv(gym.Env):

    def __init__(self, gui):

        simulator.connect_headless(gui=gui)
        simulator.setup()

        self.epi = 0
        self.high_steps = 0
        self.low_steps = 0
        self.num_task = 11

        self.loadEnv()
        self.addtionalSetting()
        self.camera_setting()
        self.index()
        self.action_definition()

        self.eefID = END_EFFECTOR_INDEX  # ee_link
        self.xyz_action_scale = XYZ_SCALE
        self.abc_action_scale = DEG_SCALE
        self.neutral_gripper_open = True
        self.is_gripper_open = True
        self.use_neutral_action = True
        self.ee_pos_high = (0.75,  0.5, 0.35)
        self.ee_pos_low =  (0.0, -0.4, 0.01)

        self.num_sim_steps = 120
        self.num_sim_steps_reset = 50
        self.num_sim_steps_discrete_action = 120

        self.high_observation_space = gym.spaces.Box(0, 255, (6,240,240))
        self.high_action_space = gym.spaces.Box(0, 1.0, (self.num_task,))
        self.low_observation_space = gym.spaces.Box(0, 255, (3,240,240))
        self.low_action_space = gym.spaces.Box(-ACTION_RANGE, ACTION_RANGE, (6,))
        self.hybrid_state_space = gym.spaces.Box(-3, 3, (6,))

        self._max_episode_steps = 40
        self.RESET_JOINT_VALUES = np.array(RESET_JOINT_VALUES)
        self.RESET_JOINT_VALUES_GRIPPER_CLOSED = np.array(RESET_JOINT_VALUES_GRIPPER_CLOSED)

    def action_definition(self):

        self.action_dict={"0" : "Laptop close",
                    "1"  : "Drawer open",
                    "2"  : "Drawer close",
                    "3"  : "Box open",
                    "4"  : "Box close",
                    "5"  : "Box push",
                    "6"  : "Green to space",
                    "7"  : "Green to drawer",
                    "8"  : "Orange to space",
                    "9"  : "Orange to box",
                    "10" : "No action left"}

    def loadEnv(self):		

        # Basic setting
        self.planeId = objects.plane()
        self.wall1Id = objects.wall1()
        self.wall2Id = objects.wall2()
        self.wall3Id = objects.wall3()
        self.tableId = objects.table()

        # Fixed object
        self.drawerId = objects.drawer()
        self.laptopId = objects.laptop()
        self.boxId = objects.box()
        self.coverId = objects.cover()
        self.trayId = objects.tray()

        # Target object
        self.greenId = objects.green()
        self.orangeId = objects.orange()

        print("----------------------------------------")
        print("Loading Panda robot ")
        self.robotId = robots.Panda() 
        self.JOINT_LIMIT_LOWER, self.JOINT_LIMIT_UPPER, self.JOINT_RANGE, self.joint_indices = robots.setup_joint_control(self.robotId)
        self.GRIPPER_LIMITS_LOW = self.JOINT_LIMIT_LOWER[-2:]
        self.GRIPPER_LIMITS_HIGH = self.JOINT_LIMIT_UPPER[-2:]
        self.GRIPPER_OPEN_STATE = [0.04, 0.04]
        self.GRIPPER_CLOSED_STATE = [0.01, 0.01]

        print(f"Robot id: {self.robotId}")
        self.done_list = [self.d_act0, self.d_act1,
                        self.d_act2, self.d_act3, self.d_act4, self.d_act5,
                        self.d_act6, self.d_act7, self.d_act8, self.d_act9]

    ############################ Init process ############################

    def reset(self, epi=0):

        self.epi +=1		
        self.high_steps = 0
        # self.cur_point = 0

        self.reset_robot()

        self.goal_object_evaluate(epi)
        self.object_rearrange()		
        self.goal_img = self.get_image()

        self.reset_object()
        self.object_rearrange()
        self.init_img = self.get_image()

        simulator.step_simulation(self.num_sim_steps_reset)

        high_o, _, _ = self.get_high_state()

        return high_o

    def object_rearrange(self):

        objects.reset_object(self.drawerId, self.drawer_pos ,self.drawerStartOrientation)
        objects.reset_object(self.laptopId, self.laptop_pos ,self.laptopStartOrientation)
        objects.reset_object(self.boxId, self.box_pos ,self.boxStartOrientation)
        objects.reset_object(self.coverId, self.cover_pos ,self.coverStartOrientation)
        objects.reset_object(self.trayId, self.tray_pos ,self.trayStartOrientation)
        objects.reset_object(self.greenId, self.green_pos ,self.greenStartOrientation)
        objects.reset_object(self.orangeId, self.orange_pos ,self.orangeStartOrientation)
        
        for i in range(50):
            p.stepSimulation()

    ############################ Step process ############################

    def high_step(self, task_num):

        # print("Action:", task_num, self.action_dict[str(task_num)])

        self.task_move(task_num)
        self.high_steps +=1
        self.low_steps = 0
        
        high_o, high_r, high_d = self.get_high_state(task_num)

        return high_o, high_r, high_d

    def low_step(self, task_num, low_a):

        self.robot_move(low_a)
        self.low_steps +=1
        low_next_o, low_r, low_d = self.get_low_state(task_num=task_num, action=low_a)

        return low_next_o, low_r, low_d

    ############################ State process ###########################

    def get_high_state(self, task_num=10):	

        image = self.get_image()
        high_done = True if task_num == 10 else False
        high_reward = 1.0 if task_num == 10 else 0.0

        return {'image': image, 'goal': self.goal_img}, high_reward, high_done

    def get_low_state(self, task_num=10, action=None):

        image = self.get_image()

        # Define states of each objects
        self.ee_position, self.ee_orientation, self.ee_xyz_vel, _ = robots.get_link_state(self.robotId,  self.eefID)
        self.gripper_state, _, self.gripper_length = robots.get_gripper_state(self.robotId, self.joint_indices)

        self.grip = np.zeros((2,))
        self.grip[int(self.is_gripper_open)] = 1

        robot_state = np.concatenate((self.ee_position,(self.ee_orientation[2],),self.grip))
        robot_state = np.round(robot_state,3)

        task_state = np.zeros((self.num_task,))
        task_state[task_num] = 1

        reward = 0.0
        done = False 
        if task_num != 10:		
            task_done = self.done_list[task_num]()
            if task_done and action[5] > 0.5:
                reward = 1.0
                done = True
                # print("OH YEAH")
    
        return {'image': image, 'hybrid': robot_state, 'task': task_state}, reward, done

    ############################ Done process ############################

    # Laptop
    def d_act0(self): 
        laptop_angle, _ = objects.get_object_joint_info(self.laptopId, 0)
        done = True if laptop_angle < 0.1 else False
        return done

    # Drawer
    def d_act1(self): 
        drawer_dist, _ = objects.get_object_joint_info(self.drawerId, 4)
        done = True if drawer_dist < -0.12 else False
        return done

    def d_act2(self): 
        drawer_dist, _ = objects.get_object_joint_info(self.drawerId, 4)
        done = True if drawer_dist > -0.02 else False
        return done

    # Box
    def d_act3(self):
        cover_pos, cover_ori = objects.get_object_link_info(self.coverId, 9)
        now_open_dist = np.linalg.norm(self.cover_open_pos-cover_pos, axis=-1)
        now_open_ori = self.cover_open_ori - cover_ori
        done = True if (now_open_dist < 0.07 and now_open_ori<0.2 and self.is_gripper_open != 0) else False
        return done

    def d_act4(self):        
        all_contact = 0
        for i in range(4):
            contact1 = np.array(p.getContactPoints(bodyA=self.coverId,bodyB=self.boxId,linkIndexA=0,linkIndexB=2*i))
            contact2 = np.array(p.getContactPoints(bodyA=self.coverId,bodyB=self.boxId,linkIndexA=0,linkIndexB=2*i+1))
            all_contact += 1 if contact1.size != 0 or contact2.size != 0 else 0
            # contact = contact1.size != 0 or contact2.size != 0        
        done = True if all_contact>0 else False
        return done

    def d_act5(self):
        box_pos, box_ori = objects.get_object_position(self.boxId)
        done = True if box_pos[0]>0.65 and abs(box_ori)<0.1 else False
        return done

    # Green
    def d_act6(self):
        contact = p.getContactPoints(bodyA=self.trayId,bodyB=self.greenId,linkIndexA=-1,linkIndexB=-1)
        done = True if (contact and self.is_gripper_open) else False
        return done

    def d_act7(self): 
        contact = p.getContactPoints(bodyA=self.drawerId,bodyB=self.greenId,linkIndexA=4,linkIndexB=-1)
        done = True if (contact and self.is_gripper_open) else False
        return done

    # Orange
    def d_act8(self):
        contact = p.getContactPoints(bodyA=self.trayId,bodyB=self.orangeId,linkIndexA=-1,linkIndexB=-1)
        done = True if (contact and self.is_gripper_open) else False
        return done

    def d_act9(self):
        contact = p.getContactPoints(bodyA=self.boxId,bodyB=self.orangeId,linkIndexA=-1,linkIndexB=-1)
        done = True if (contact and self.is_gripper_open) else False
        return done
    
    ############################ Move process ###########################

    def robot_move(self, action):

        xyz_action = action[:3]
        yaw_action = action[3]
        gripper_action = action[4]
        neutral_action = action[5]

        ee_pos, ee_rad, _, _ = robots.get_link_state(
            self.robotId, self.eefID)
        joint_states, _ = robots.get_joint_states(self.robotId,
                                                  self.joint_indices)
        gripper_state = np.asarray([joint_states[-2], joint_states[-1]])

        target_ee_pos = ee_pos + self.xyz_action_scale * xyz_action
        ee_rad[0] = 3.141592
        ee_rad[1] = 0.0
        ee_deg = robots.rad_to_deg(ee_rad)
        ee_deg[2] += self.abc_action_scale * yaw_action
        target_ee_deg = ee_deg
        target_ee_quat = robots.deg_to_quat(target_ee_deg)

        if gripper_action > 0.5 and not self.is_gripper_open:
            num_sim_steps = self.num_sim_steps_discrete_action
            target_gripper_state = self.GRIPPER_OPEN_STATE
            self.is_gripper_open = True  # TODO(avi): Clean this up

        elif gripper_action < -0.5 and self.is_gripper_open:
            num_sim_steps = self.num_sim_steps_discrete_action
            target_gripper_state = self.GRIPPER_CLOSED_STATE
            self.is_gripper_open = False  # TODO(avi): Clean this up
        else:
            num_sim_steps = self.num_sim_steps
            if self.is_gripper_open:
                target_gripper_state = self.GRIPPER_OPEN_STATE
            else:
                target_gripper_state = self.GRIPPER_CLOSED_STATE

        
        target_ee_pos = np.clip(target_ee_pos, self.ee_pos_low,
                                self.ee_pos_high)
        target_gripper_state = np.clip(target_gripper_state, self.GRIPPER_LIMITS_LOW,
                                       self.GRIPPER_LIMITS_HIGH)

        robots.apply_action_ik(
            target_ee_pos, target_ee_quat, target_gripper_state,
            self.robotId,
            self.eefID, self.joint_indices,
            lower_limit=self.JOINT_LIMIT_LOWER,
            upper_limit=self.JOINT_LIMIT_UPPER,
            rest_pose=self.RESET_JOINT_VALUES,
            joint_range=self.JOINT_RANGE,
            num_sim_steps=num_sim_steps)

        # neutral_action = 0.0
        if self.use_neutral_action and neutral_action > 0.5:
            if self.neutral_gripper_open:
                robots.move_to_neutral(
                    self.robotId,
                    self.joint_indices,
                    self.RESET_JOINT_VALUES)
                self.is_gripper_open = True
            else:
                robots.move_to_neutral(
                    self.robotId,
                    self.joint_indices,
                    self.RESET_JOINT_VALUES_GRIPPER_CLOSED)
                self.is_gripper_open = False

    def task_move(self, task_num):

        self.reset_robot()

        # Laptop manipulataion
        if task_num == 0:
            laptop_angle = 0.1
            p.resetJointState(self.laptopId, 0, laptop_angle)
            self.laptop_open = False
        
        # Drawer manipulation
        elif task_num == 1:
            drawer_joint = random.uniform(-0.185, -0.178)
            p.resetJointState(self.drawerId, 4, drawer_joint)
            self.drawer_open = True
        elif task_num == 2:
            drawer_joint = random.uniform(-0.01, 0.005)
            p.resetJointState(self.drawerId, 4, drawer_joint)
            self.drawer_open = False
            if self.obj_in_drawer:
                p.stepSimulation()
                drawer_put_pos, drawer_put_ori = objects.get_object_link_info(self.drawerId, 4)
                drawer_put_pos += np.array([random.uniform(-0.02, 0.02),random.uniform(-0.02, 0.02),0.03])
                drawer_put_ori += random.uniform(-0.3, 0.3)
                self.greenStartOrientation = p.getQuaternionFromEuler([0 ,0, drawer_put_ori])
                objects.reset_object(self.greenId,  drawer_put_pos, self.greenStartOrientation)

        # Box manipulation
        elif task_num == 3:
            self.coverStartOrientation = p.getQuaternionFromEuler([0 ,0, self.cover_open_ori])
            objects.reset_object(self.coverId,  self.cover_open_pos,self.coverStartOrientation)		
            self.box_opened = True
        elif task_num == 4:
            box_put_pos, box_put_ori = objects.get_object_position(self.boxId)
            box_put_pos[2] += 0.1
            self.coverStartOrientation = p.getQuaternionFromEuler([0 ,0, box_put_ori])
            objects.reset_object(self.coverId,  box_put_pos, self.coverStartOrientation)		
            self.box_opened = False
        elif task_num == 5:
            box_push_pos, box_push_ori = objects.get_object_position(self.boxId)
            box_push_pos[0] = 0.7
            self.boxStartOrientation = p.getQuaternionFromEuler([0 ,0, box_push_ori])
            objects.reset_object(self.boxId,  box_push_pos, self.boxStartOrientation)
            cover_push_pos = box_push_pos.copy()
            cover_push_pos[2] += 0.1
            self.coverStartOrientation = p.getQuaternionFromEuler([0 ,0, box_push_ori])
            objects.reset_object(self.coverId,  cover_push_pos, self.coverStartOrientation)            
            if self.obj_in_box:
                obj_push_pos = box_push_pos.copy()
                obj_push_pos[2] += 0.03
                objects.reset_object(self.orangeId,  obj_push_pos, self.boxStartOrientation)           
            self.box_pushed = True            
        
        # Green manipulation
        elif task_num == 6:
            tray_pos, tray_ori = objects.get_object_position(self.trayId)
            tray_pos += np.array([random.uniform(-0.02, 0.02),random.uniform(-0.025, 0.025),0.03])
            tray_ori += random.uniform(-0.2, 0.2)
            self.greenStartOrientation = p.getQuaternionFromEuler([0 ,0, tray_ori])
            objects.reset_object(self.greenId,  tray_pos ,self.greenStartOrientation)		
            self.obj_on_tray = True
            self.obj_on_laptop = False
            self.green_cur_location = 1
            self.green_done = (self.green_cur_location == self.green_goal_location)
        elif task_num == 7:
            drawer_put_pos, drawer_put_ori = objects.get_object_link_info(self.drawerId, 4)
            drawer_put_pos += np.array([random.uniform(-0.03, 0.03),random.uniform(-0.03, 0.03),0.03])
            drawer_put_ori += random.uniform(-0.4, 0.4)
            self.greenStartOrientation = p.getQuaternionFromEuler([0 ,0, drawer_put_ori])
            objects.reset_object(self.greenId,  drawer_put_pos, self.greenStartOrientation)
            self.obj_on_tray = False
            self.obj_in_drawer = True
            self.green_cur_location = 2
            self.green_done = (self.green_cur_location == self.green_goal_location)

        # Orange manipulation
        elif task_num == 8:
            tray_pos, tray_ori = objects.get_object_position(self.trayId)
            tray_pos += np.array([random.uniform(-0.02, 0.02),random.uniform(-0.025, 0.025),0.03])
            tray_ori += random.uniform(-0.2, 0.2)
            self.orangeStartOrientation = p.getQuaternionFromEuler([0 ,0, tray_ori])
            objects.reset_object(self.orangeId,  tray_pos ,self.orangeStartOrientation)		
            self.obj_on_tray = True
            self.obj_on_laptop = False
            self.orange_cur_location = 1
            self.orange_done = (self.orange_cur_location == self.orange_goal_location)
        elif task_num == 9:
            box_put_pos, box_put_ori = objects.get_object_position(self.boxId)
            box_put_pos += np.array([random.uniform(-0.02, 0.02),random.uniform(-0.02, 0.02),0.03])
            box_put_ori += random.uniform(-0.4, 0.4)
            self.orangeStartOrientation = p.getQuaternionFromEuler([0 ,0, box_put_ori])
            objects.reset_object(self.orangeId,  box_put_pos, self.orangeStartOrientation)
            self.obj_on_tray = False
            self.obj_in_box = True
            self.orange_cur_location = 2
            self.orange_done = (self.orange_cur_location == self.orange_goal_location)

        self.obj_section = [self.green_cur_location, self.orange_cur_location]

        for _ in range(50):
            p.stepSimulation()

    ############################ Image process ##########################
    
    def camera_setting(self):

        # Camera setting
        self.width = img_size
        self.height = img_size

        fov = 59
        aspect = 1280.0/720.0
        near = 0.01
        far = 1.1

        self.cam_pos = np.array([-0.032,  0.208,  0.368])
        self.cam_tar = np.array([ 0.288,  0.208,  0.203])
        self.cam_up  = np.array([ 0.001,  0.000,  0.001])
                
        self.view_matrix  = p.computeViewMatrix(self.cam_pos, self.cam_tar, self.cam_up)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov,aspect,near,far)

    def index(self):
        self.robot_index = []
        self.table_index = []
        self.wall_index = []
        self.laptop_index = []
        self.drawer_index = []
        self.tray_index = []
        self.box_index = []
        self.cover_index = []
        self.obj_index = []

        for i in range(21):
            self.robot_index.append(self.robotId+((i)<<24))

        for i in range(2):
            self.table_index.append(self.tableId+((i)<<24))

        for i in range(2):
            self.wall_index.append(self.wall1Id+((i)<<24))
            self.wall_index.append(self.wall2Id+((i)<<24))
            self.wall_index.append(self.wall3Id+((i)<<24))

        for i in range(3):
            self.laptop_index.append(self.laptopId+((i)<<24))

        for i in range(15):
            self.drawer_index.append(self.drawerId+((i)<<24))

        for i in range(7):
            self.tray_index.append(self.trayId+((i)<<24))

        for i in range(12):
            self.box_index.append(self.boxId+((i)<<24))

        for i in range(13):
            self.cover_index.append(self.coverId+((i)<<24))

        for i in range(2):
            self.obj_index.append(self.greenId+((i)<<24))            
            self.obj_index.append(self.orangeId+((i)<<24))

    def get_image(self):

        direction = np.array([-1.5,0,4.5])
        Color=np.array([0.8,0.8,0.8])
        AmbientCoeff=0.5
        DiffuseCoeff=0.5
        SpecularCoeff=0.02

        rand_pos = np.random.uniform(-0.025, 0.025, 3)
        rand_tar = np.random.uniform(-0.025, 0.025, 3)
        rand_tar[1] = rand_pos[1]

        self.view_matrix  = p.computeViewMatrix(self.cam_pos+rand_pos, self.cam_tar+rand_tar, self.cam_up)

        images = p.getCameraImage(self.width,self.height,self.view_matrix,self.projection_matrix, shadow=True, lightColor=Color, lightAmbientCoeff=AmbientCoeff, lightDirection=direction, lightDiffuseCoeff=DiffuseCoeff, lightSpecularCoeff=SpecularCoeff)
        rgb = np.reshape(images[2], (self.height, self.width, 4))[:,:,:3]

        self.rgb = rgb[6:6+240,12:12+240,:]
        # plt.imsave(f"sample/{self.steps}.png",self.rgb)
        image = np.transpose(self.rgb,[2,0,1]).astype(np.uint8)

        return image

    def dep_get(self,dep):
        # Depth process
        min_dep = MIN_DEP
        max_dep = MAX_DEP
        depth = max_dep * min_dep / (max_dep - (max_dep - min_dep) * dep)
        dep_img = (255*((depth-min_dep)/(max_dep - min_dep)))
        dep_img = np.reshape(dep_img, (self.height, self.width, 1)).astype(np.uint8)

        return dep_img.astype(np.uint8)

    def seg_dep_get(self, seg, dep):

        # Segmentation process 
        seg_img = np.zeros((self.height, self.width, 9))

        for ind in self.robot_index:
            seg_img[seg == ind,0] = 255

        for ind in self.table_index:
            seg_img[seg == ind,1] = 255
        
        for ind in self.wall_index:
            seg_img[seg == ind,2] = 255
        
        for ind in self.laptop_index:
            seg_img[seg == ind,3] = 255
        
        for ind in self.drawer_index:
            seg_img[seg == ind,4] = 255

        for ind in self.tray_index:
            seg_img[seg == ind,5] = 255

        for ind in self.box_index:
            seg_img[seg == ind,6] = 255

        for ind in self.cover_index:
            seg_img[seg == ind,7] = 255

        for ind in self.obj_index:
            seg_img[seg == ind,8] = 255

        # Depth process
        min_dep = MIN_DEP
        max_dep = MAX_DEP
        depth = max_dep * min_dep / (max_dep - (max_dep - min_dep) * dep)
        dep_img = (255*((depth-min_dep)/(max_dep - min_dep)))
        dep_img = np.reshape(dep_img, (self.height, self.width, 1)).astype(np.uint8)

        return seg_img.astype(np.uint8), dep_img.astype(np.uint8)
    
    def canonical_color_definition(self):
    
        self.ROBOT_COLOR = [ [0.1, 0.1, 0.9, 1], [0.9, 0.1, 0.1, 1], [0.9, 0.9, 0.1, 1],[0.1, 0.9, 0.1, 1]]
        self.TABLE_COLOR = [0.937, 0.729, 0.494, 1]
        self.WALL_COLOR = [0.9, 0.9, 0.9, 1]

        self.LAPTOP_COLOR = [1.0, 1.0, 1.0, 1.0]
        self.DRAWER_OUT_COLOR = [0.0, 0.522, 0.565, 1.0]
        self.DRAWER_IN_COLOR = [0.792, 0.047, 0.047, 1.0]
        self.DRAWER_HANDLE_COLOR = [0.879, 0.719, 0.125, 1.0]
        self.TRAY_COLOR = [0.722, 0.757, 0.780, 1.0]
        self.INNER_COLOR = [0.9, 0.9, 0.9, 1]
        self.BOX_OUT_COLOR = [0.106, 0.463, 0.720, 1]
        self.COVER_OUT_COLOR = [0.957, 0.443, 0.710, 1]
        self.COVER_HANDLE_COLOR = [0.973, 0.741, 0.067, 1]
        self.GREEN_COLOR = [0.753, 0.847, 0.0, 1.0]
        self.ORANGE_COLOR = [0.831, 0.271, 0.094, 1.0]
        
    def canonical_color(self):
        
        for i in range(11):
            if i > 7:
                color = self.ROBOT_COLOR[(i-1)%len(self.ROBOT_COLOR)]
            else:
                color = self.ROBOT_COLOR[(i)%len(self.ROBOT_COLOR)]
            if i ==10:
                color = self.ROBOT_COLOR[(i-2)%len(self.ROBOT_COLOR)]
            p.changeVisualShape(self.robotId, i, rgbaColor=color)

        # Wall
        p.changeVisualShape(self.wall1Id,-1, rgbaColor=self.WALL_COLOR)
        p.changeVisualShape(self.wall2Id,-1, rgbaColor=self.WALL_COLOR)
        p.changeVisualShape(self.wall3Id,-1, rgbaColor=self.WALL_COLOR)
        # Table
        p.changeVisualShape(self.tableId,-1, rgbaColor=self.TABLE_COLOR)
        # Laptop
        for i in range(2):
            p.changeVisualShape(self.laptopId,i-1, rgbaColor=self.LAPTOP_COLOR)
        # Drawer_out_color
        for i in self.drawer_out_list:            
            p.changeVisualShape(self.drawerId, i, rgbaColor=self.DRAWER_OUT_COLOR)
        # Drawer_in_color
        for i in self.drawer_inner:            
            p.changeVisualShape(self.drawerId, i, rgbaColor=self.DRAWER_IN_COLOR)
        # Drawer_handle_color
        for i in self.drawer_handle_list:            
            p.changeVisualShape(self.drawerId, i, rgbaColor=self.DRAWER_HANDLE_COLOR)
        # Tray_color
        for i in range(6):            
            p.changeVisualShape(self.trayId, i-1, rgbaColor=self.TRAY_COLOR)
        # Inner_color
        for i in self.box_inner:            
            p.changeVisualShape(self.boxId,i, rgbaColor=self.INNER_COLOR)
        for i in self.cover_inner:          
            p.changeVisualShape(self.coverId,i, rgbaColor=self.INNER_COLOR)
        # Box_outer_color
        for i in self.box_out_list:
            p.changeVisualShape(self.boxId,i, rgbaColor=self.BOX_OUT_COLOR)
        # Cover_outer_color
        for i in self.cover_out_list:
            p.changeVisualShape(self.coverId,i, rgbaColor=self.COVER_OUT_COLOR)
        # Cover_handle_color
        p.changeVisualShape(self.coverId,9, rgbaColor=self.COVER_HANDLE_COLOR)
        # Green_color
        p.changeVisualShape(self.greenId,-1, rgbaColor=self.GREEN_COLOR)
        # Orange_color
        p.changeVisualShape(self.orangeId,-1, rgbaColor=self.ORANGE_COLOR)

    def addtionalSetting(self):
        p.setCollisionFilterPair(bodyUniqueIdA=self.boxId,bodyUniqueIdB=self.orangeId,linkIndexA=8,linkIndexB=-1,enableCollision=0)
        p.setJointMotorControl2(self.drawerId, 4, p.VELOCITY_CONTROL, force=5)
        p.setJointMotorControl2(self.laptopId, 0, p.VELOCITY_CONTROL, force=3)
        
        self.box_inner = [-1,0,2,4,6]
        self.cover_inner = [-1,1,3,5,7]
        self.drawer_inner = [4,5,6,7,8]
        self.box_out_list = [1,3,5,7]
        self.cover_out_list = [0,2,4,6,8]
        self.drawer_out_list = [-1,0,1,2,3]
        self.drawer_handle_list = [9,10,11]
        
        self.canonical_color_definition()
        self.canonical_color()

    ############################ Env process ############################

    def reset_robot(self):
        robots.reset_robot(self.robotId, self.joint_indices, self.RESET_JOINT_VALUES)
        self.is_gripper_open = True

    def reset_object(self):
    
        self.obj_in_drawer = False
        self.obj_in_box = False
        self.obj_on_laptop = False
        self.obj_on_tray = False
        
        self.box_opened = False
        self.box_pushed = False        
                
        self.green_done = False
        self.orange_done = False
        
        # Laptop reset
        self.laptop_pos = np.array([0.45+random.uniform(-0.05, 0.015),0.41+random.uniform(-0.04, 0.03),0.0])
        self.laptop_ori = random.uniform(-0.05, 0.05)#  + 3.141592/2
        self.laptopStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.laptop_ori])
        self.laptop_open = True
        p.resetJointState(self.laptopId, 0, self.laptop_reset_joint)
        
        # Drawer reset
        self.drawer_pos = np.array([0.70+random.uniform(-0.015, 0.04),0.38+random.uniform(-0.05, 0.03),-0.002])
        self.drawer_ori = math.pi/2.0 + random.uniform(-0.05, 0.05) # + 3.141592/2
        self.drawerStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.drawer_ori])
        self.drawer_joint = random.uniform(-0.03, 0.01)
        self.drawer_open = False
        p.resetJointState(self.drawerId, 4, self.drawer_joint)

        # Box reset
        self.box_ori = random.uniform(-0.05, 0.05)
        self.box_reset_pos = np.array([0.48+random.uniform(-0.03, 0.03),-0.15+random.uniform(-0.03, 0.03),0.0])
        self.boxStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.box_ori])
        
        # Cover reset
        self.box_pos = self.box_reset_pos
        self.cover_pos = self.box_pos.copy()
        self.cover_pos[2] += 0.043
        self.cover_ori = self.box_ori
        self.coverStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.cover_ori])
       
        # Green pose
        self.object_loaction = [0,1]
        random.shuffle(self.object_loaction)
        self.green_cur_location = self.object_loaction[0]
        self.orange_cur_location = self.object_loaction[1]

        if self.orange_goal_location == 2  and self.green_cur_location == 1:
            self.green_cur_location, self.orange_cur_location = 0, 1
        if self.green_goal_location == 2  and self.orange_cur_location == 1:
            self.green_cur_location, self.orange_cur_location = 1, 0

        if self.remove_obj==0:
            self.orange_cur_location = 0
        if self.remove_obj==1:
            self.green_cur_location = 0        
            
        if self.green_goal_location == 0:
            self.green_pos = np.array([0.0, 0.0, -0.4])
            self.green_ori = 0.0
        else:
            if self.green_cur_location == 0:
                # On the laptop
                self.green_pos = self.laptop_pos.copy()
                self.green_pos += np.array([random.uniform(-0.04, 0.04), random.uniform(-0.035, 0.035), 0.05])
                self.green_ori = random.uniform(-0.1, 0.1)
                self.greenStartOrientation = p.getQuaternionFromEuler([0 ,0, self.green_ori])
                self.obj_on_laptop = True
            else:
                # On the tray
                self.green_pos = self.tray_pos.copy()
                self.green_pos += np.array([random.uniform(-0.03, 0.03), random.uniform(-0.035, 0.035), 0.05])
                self.green_ori = random.uniform(-0.2, 0.2)
                self.greenStartOrientation = p.getQuaternionFromEuler([0 ,0, self.green_ori])
                self.obj_on_tray = True                               
        if (self.green_goal_location == self.green_cur_location) or (not self.green_goal_location):
            self.green_done = True         
        
        # Orange pose
        if self.orange_goal_location == 0:
            self.orange_pos = np.array([0.0, 0.0, -0.4])
            self.orange_ori = 0.0
        else:
            if self.orange_cur_location == 0:
                # On the laptop
                self.orange_pos = self.laptop_pos.copy()
                self.orange_pos += np.array([random.uniform(-0.04, 0.04), random.uniform(-0.035, 0.035), 0.05])
                self.orange_ori = random.uniform(-0.1, 0.1)
                self.orangeStartOrientation = p.getQuaternionFromEuler([0 ,0, self.orange_ori])
                self.obj_on_laptop = True
            else:
                # On the tary
                self.orange_pos = self.tray_pos.copy()
                self.orange_pos += np.array([random.uniform(-0.03, 0.03), random.uniform(-0.035, 0.035), 0.05])
                self.orange_ori = random.uniform(-0.2, 0.2)
                self.orangeStartOrientation = p.getQuaternionFromEuler([0 ,0, self.orange_ori])
                self.obj_on_tray = True                  
            
        if (self.orange_goal_location == self.orange_cur_location) or (not self.orange_goal_location):
            self.orange_done = True 
                

        self.obj_cur_location = [self.green_cur_location, self.orange_cur_location]

    def goal_object(self):
    
        # Laptop pose
        self.laptop_goal = random.randint(0,1)        
        self.laptop_pos = np.array([0.45+random.uniform(-0.05, 0.015),0.41+random.uniform(-0.04, 0.03),0.0])
        self.laptop_ori = random.uniform(-0.05, 0.05)  #+ 3.141592/2
        self.laptopStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.laptop_ori])
        if self.laptop_goal == 0:
            self.laptop_joint = 0.0
            self.laptop_reset_joint = random.uniform(1.2, 1.5)
            p.resetJointState(self.laptopId, 0, self.laptop_joint)
        else:
            self.laptop_joint = random.uniform(1.2, 1.5)
            self.laptop_reset_joint = self.laptop_joint
            p.resetJointState(self.laptopId, 0, self.laptop_joint)
        
        
        # Drawer pose
        self.drawer_pos = np.array([0.70+random.uniform(-0.015, 0.04),0.38+random.uniform(-0.05, 0.03), -0.002])
        self.drawer_ori = math.pi/2.0 + random.uniform(-0.05, 0.05) # + 3.141592/2
        self.drawerStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.drawer_ori])
        self.drawer_joint = 0.0
        p.resetJointState(self.drawerId, 4, self.drawer_joint)
        
        # Box pose
        self.box_goal_location = random.randint(0,1)
        if self.box_goal_location == 0:
            self.box_pos = np.array([0.48+random.uniform(-0.03, 0.04),-0.15+random.uniform(-0.03, 0.03),0.0]) #+random.uniform(-0.03, 0.03)
            self.box_ori = random.uniform(-0.05, 0.05)
            self.boxStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.box_ori])
            self.box_reset_pos = self.box_pos
        else:
            self.box_pos = np.array([0.7+random.uniform(-0.03, 0.04),-0.15+random.uniform(-0.03, 0.03),0.0]) #+random.uniform(-0.03, 0.03)
            self.box_ori = random.uniform(-0.05, 0.05)
            self.box_reset_pos = np.array([0.48,-0.15+random.uniform(-0.03, 0.03),0.0])
            self.boxStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.box_ori])
        
        # Cover pose
        self.cover_pos = self.box_pos.copy()
        self.cover_pos[2] += 0.1
        self.cover_ori = self.box_ori
        self.coverStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.cover_ori])
        self.cover_open_pos, self.cover_open_ori = np.array([0.7+random.uniform(-0.03, 0.03), -0.15+random.uniform(-0.03, 0.03), 0.07]), 0.0
        
        # Tray pose
        self.tray_pos = np.array([0.43+random.uniform(-0.025, 0.025),0.09+random.uniform(-0.025, 0.025),0.001])
        self.tray_ori = random.uniform(-0.05, 0.05)
        self.trayStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.tray_ori])
        
        self.remove_obj = random.randint(0,3)
        self.obj_in_tray = random.randint(0,3)
        
        
        # Green pose
        if self.remove_obj == 0:
            self.green_pos = np.array([0.0, 0.0, -0.4])
            self.green_ori = 0.0
            self.greenStartOrientation = p.getQuaternionFromEuler([0 ,0, self.green_ori])
            self.green_goal_location = 0
        elif self.obj_in_tray == 0:
            # On the Tray
            self.green_pos = self.tray_pos.copy()
            self.green_pos[2] += 0.03
            self.green_ori = 0.0
            self.greenStartOrientation = p.getQuaternionFromEuler([0 ,0, self.green_ori])
            self.green_goal_location = 1
        else:
            # In the Drawer
            self.green_pos = self.drawer_pos.copy()
            self.green_pos[2] += 0.03
            self.green_ori = self.drawer_ori - math.pi/2.0
            self.greenStartOrientation = p.getQuaternionFromEuler([0 ,0, self.green_ori])
            self.green_goal_location = 2		

        self.green_goal_pos = self.green_pos.copy()
        self.green_goal_ori = self.green_ori
        
        # Orange pose
        if self.remove_obj == 1:
            self.orange_pos = np.array([0.0, 0.0, -0.4])
            self.orange_ori = 0.0
            self.orangeStartOrientation = p.getQuaternionFromEuler([0 ,0, self.orange_ori])
            self.orange_goal_location = 0
        elif self.obj_in_tray == 1:
            # On the Tray
            self.orange_pos = self.tray_pos.copy()
            self.orange_pos[2] += 0.03
            self.orange_ori = 0.0
            self.orangeStartOrientation = p.getQuaternionFromEuler([0 ,0, self.orange_ori])
            self.orange_goal_location = 1
        else:
            # In the Box
            self.orange_pos = self.box_pos.copy()
            self.orange_pos[2] += 0.03
            self.orange_ori = self.box_ori
            self.orangeStartOrientation = p.getQuaternionFromEuler([0 ,0, self.orange_ori])
            self.orange_goal_location = 2	

        self.orange_goal_pos = self.orange_pos.copy()
        self.orange_goal_ori = self.orange_ori


        self.obj_goal_location = [self.green_goal_location, self.orange_goal_location]
        
    def goal_object_evaluate(self, epi): 
        
        if epi == 0:
            laptop_close=True
            box_push=True
            green_location=1
            orange_location=2 
        elif epi == 1:
            laptop_close=True
            box_push=False
            green_location=2
            orange_location=1
        else:
            laptop_close=False
            box_push=True
            green_location=2
            orange_location=2

        # Laptop pose
        self.laptop_goal = laptop_close     
        self.laptop_pos = np.array([0.45+random.uniform(-0.05, 0.015),0.41+random.uniform(-0.04, 0.03),0.0])
        self.laptop_ori = random.uniform(-0.05, 0.05)  #+ 3.141592/2
        self.laptopStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.laptop_ori])
        if self.laptop_goal == 1:
            self.laptop_joint = 0.0
            self.laptop_reset_joint = random.uniform(1.2, 1.5)
            p.resetJointState(self.laptopId, 0, self.laptop_joint)
        else:
            self.laptop_joint = random.uniform(1.2, 1.5)
            self.laptop_reset_joint = self.laptop_joint
            p.resetJointState(self.laptopId, 0, self.laptop_joint)
        
        
        # Drawer pose
        self.drawer_pos = np.array([0.70+random.uniform(-0.015, 0.04),0.38+random.uniform(-0.05, 0.03), -0.002])
        self.drawer_ori = math.pi/2.0 + random.uniform(-0.05, 0.05) # + 3.141592/2
        self.drawerStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.drawer_ori])
        self.drawer_joint = 0.0
        p.resetJointState(self.drawerId, 4, self.drawer_joint)
        
        # Box pose
        self.box_goal_location = box_push
        if self.box_goal_location == 0:
            self.box_pos = np.array([0.48+random.uniform(-0.03, 0.04),-0.15+random.uniform(-0.03, 0.03),0.0]) #+random.uniform(-0.03, 0.03)
            self.box_ori = random.uniform(-0.05, 0.05)
            self.boxStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.box_ori])
            self.box_reset_pos = self.box_pos
        else:
            self.box_pos = np.array([0.7+random.uniform(-0.03, 0.04),-0.15+random.uniform(-0.03, 0.03),0.0]) #+random.uniform(-0.03, 0.03)
            self.box_ori = random.uniform(-0.05, 0.05)
            self.box_reset_pos = np.array([0.48,-0.15+random.uniform(-0.03, 0.03),0.0])
            self.boxStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.box_ori])
        
        # Cover pose
        self.cover_pos = self.box_pos.copy()
        self.cover_pos[2] += 0.1
        self.cover_ori = self.box_ori
        self.coverStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.cover_ori])
        self.cover_open_pos, self.cover_open_ori = np.array([0.7+random.uniform(-0.03, 0.03), -0.15+random.uniform(-0.03, 0.03), 0.07]), 0.0
        
        # Tray pose
        self.tray_pos = np.array([0.43+random.uniform(-0.025, 0.025),0.09+random.uniform(-0.025, 0.025),0.001])
        self.tray_ori = random.uniform(-0.05, 0.05)
        self.trayStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.tray_ori])
        
        self.remove_obj = 2
        self.obj_in_tray = 2        
        
        # Green pose
        if green_location == 0:
            self.green_pos = np.array([0.0, 0.0, -0.4])
            self.green_ori = 0.0
            self.greenStartOrientation = p.getQuaternionFromEuler([0 ,0, self.green_ori])
            self.green_goal_location = 0
        elif green_location == 1:
            # On the Tray
            self.green_pos = self.tray_pos.copy()
            self.green_pos[2] += 0.03
            self.green_ori = 0.0
            self.greenStartOrientation = p.getQuaternionFromEuler([0 ,0, self.green_ori])
            self.green_goal_location = 1
            self.obj_in_tray = 0
        else:
            # In the Drawer
            self.green_pos = self.drawer_pos.copy()
            self.green_pos[2] += 0.03
            self.green_ori = self.drawer_ori - math.pi/2.0
            self.greenStartOrientation = p.getQuaternionFromEuler([0 ,0, self.green_ori])
            self.green_goal_location = 2		

        self.green_goal_pos = self.green_pos.copy()
        self.green_goal_ori = self.green_ori
        
        # Orange pose
        if orange_location == 0:
            self.orange_pos = np.array([0.0, 0.0, -0.4])
            self.orange_ori = 0.0
            self.orangeStartOrientation = p.getQuaternionFromEuler([0 ,0, self.orange_ori])
            self.orange_goal_location = 0
        elif orange_location == 1:
            # On the Tray
            self.orange_pos = self.tray_pos.copy()
            self.orange_pos[2] += 0.03
            self.orange_ori = 0.0
            self.orangeStartOrientation = p.getQuaternionFromEuler([0 ,0, self.orange_ori])
            self.orange_goal_location = 1
            self.obj_in_tray = 1
        else:
            # In the Box
            self.orange_pos = self.box_pos.copy()
            self.orange_pos[2] += 0.03
            self.orange_ori = self.box_ori
            self.orangeStartOrientation = p.getQuaternionFromEuler([0 ,0, self.orange_ori])
            self.orange_goal_location = 2	

        self.orange_goal_pos = self.orange_pos.copy()
        self.orange_goal_ori = self.orange_ori


        self.obj_goal_location = [self.green_goal_location, self.orange_goal_location]

    def get_possible_action(self):

        possible_action = []
        
        # Laptop command
        if not self.obj_on_laptop and self.laptop_open and self.laptop_goal==0:
            possible_action.append(0)

        # Drawer command
        if not self.drawer_open and not self.obj_in_drawer and self.green_goal_location==2:
            possible_action.append(1)
        elif self.obj_in_drawer and self.drawer_open:
            possible_action.append(2)
        elif self.drawer_open and not self.green_goal_location==2:
            possible_action.append(2)
            
        # Box command
        if not self.box_opened and not self.obj_in_box and self.orange_goal_location==2:
            possible_action.append(3)
        elif self.obj_in_box and self.box_opened:
            possible_action.append(4)
        elif self.box_opened and not self.orange_goal_location==2:
            possible_action.append(4)
        elif not self.box_opened and self.obj_in_box and not self.box_pushed and self.box_goal_location==1:
            possible_action.append(5)
        elif not self.box_opened and self.orange_goal_location !=2 and not self.box_pushed and self.box_goal_location==1:
            possible_action.append(5)

        # Green command
        if self.green_cur_location == 0 and not self.obj_on_tray and not self.remove_obj == 0:
            possible_action.append(6)
        elif self.green_cur_location == 1 and self.drawer_open and not self.remove_obj == 0:
            possible_action.append(7)
        
        # Orange command
        if self.orange_cur_location == 0 and not self.obj_on_tray and not self.remove_obj == 1:
            possible_action.append(8)
        elif self.orange_cur_location == 1 and self.box_opened and not self.remove_obj == 1:
            possible_action.append(9)

        if not len(possible_action):
            possible_action.append(10)
        
        # print(possible_action)
        
        return np.array(possible_action)

    def robot_action_make(self, action):

        # Laptop
        laptop_pre_pos, laptop_pre_ori = objects.get_object_link_info(self.laptopId, 1)
        laptop_tar_pos, laptop_tar_ori = objects.get_object_link_info(self.laptopId, 2)
        laptop_angle, _ = objects.get_object_joint_info(self.laptopId, 0)

        # Drawer
        handle_pos, handle_ori = objects.get_object_link_info(self.drawerId, 11)
        handle_pos[2] -= 0.01
        handle_ori -= math.pi/2.0
        drawer_vector = np.array([math.cos(self.drawer_ori), math.sin(self.drawer_ori), 0.0])
        drawer_put_pos, drawer_put_ori = objects.get_object_link_info(self.drawerId, 4)
        drawer_put_pos[2] += 0.1
        drawer_put_ori -= math.pi/2.0 
        
        # Box
        box_put_pos, box_put_ori = objects.get_object_position(self.boxId)
        box_push_pos = box_put_pos.copy()
        box_put_pos[2] += 0.07
        box_push_pos[0] -= 0.18
        # box_push_pos[2] += 0.07
        
        # Cover
        cover_pos, cover_ori = objects.get_object_link_info(self.coverId, 9)
        cover_open_pos, cover_open_ori = self.cover_open_pos, self.cover_open_ori
        
        # Tray
        tray_pos, tray_ori = objects.get_object_position(self.trayId)
        tray_pos[2] += 0.05        

        # Object
        green_pos, green_ori = objects.get_object_position(self.greenId)
        orange_pos, orange_ori = objects.get_object_position(self.orangeId)

        # Laptop manipulation
        if action == 0:
            self.pre_pos, self.pre_ori = laptop_pre_pos, laptop_pre_ori
            self.tar_pos, self.tar_ori = laptop_tar_pos, laptop_tar_ori
        
        # Drawer manipulation
        elif action == 1:
            self.tar_pos, self.tar_ori = handle_pos, handle_ori
            self.tar_vec = drawer_vector
        elif action == 2:
            self.tar_pos, self.tar_ori = handle_pos, handle_ori
            self.tar_vec = drawer_vector

        # Box manipulation
        elif action == 3:
            self.tar_pos, self.tar_ori = cover_pos, cover_ori
            self.drop_pos, self.drop_ori = cover_open_pos, cover_open_ori
        elif action == 4:
            self.tar_pos, self.tar_ori = cover_pos, cover_ori
            self.drop_pos, self.drop_ori = box_put_pos, box_put_ori
        elif action == 5:
            self.tar_pos, self.tar_ori = box_push_pos, box_put_ori
            self.tar_vec = np.array([1.0, 0.0, 0.0])

        # Green manipulation
        elif action == 6:
            self.tar_pos, self.tar_ori = green_pos, green_ori
            self.drop_pos, self.drop_ori = tray_pos, tray_ori
        elif action == 7:
            self.tar_pos, self.tar_ori = green_pos, green_ori
            self.drop_pos, self.drop_ori = drawer_put_pos, drawer_put_ori

        # Orange manipulation
        elif action == 8:
            self.tar_pos, self.tar_ori = orange_pos, orange_ori
            self.drop_pos, self.drop_ori = tray_pos, tray_ori
        elif action == 9:
            self.tar_pos, self.tar_ori = orange_pos, orange_ori
            self.drop_pos, self.drop_ori = box_put_pos, box_put_ori

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
