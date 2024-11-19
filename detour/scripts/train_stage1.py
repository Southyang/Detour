from collections import deque
import datetime
import os
import logging
import sys
import numpy as np
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI
import socket

from torch.optim import Adam
from model.net import DetourPolicy
from model.ppo import ppo_update_stage1, ppo_update_stage1_improved, generate_train_data, generate_action_diffusion, \
    transform_buffer_diffusion, ppo_update_stage1_diffusion
from model.ppo import generate_action, generate_action_improved
from model.ppo import transform_buffer, transform_buffer_improved

from env_stage1 import Environment

from torch.utils.tensorboard import SummaryWriter

import cv2


MAX_EPISODES = 8000
LASER_BEAM = 512
LASER_HIST = 3
# HORIZON = 512   # 更新一次的步数，如果150 step算超时，那么这里取128
HORIZON = 512
GAMMA = 0.99
LAMDA = 0.95
# BATCH_SIZE = 1024
BATCH_SIZE = 32
EPOCH = 2
COEFF_ENTROPY = 5e-4     # 5e-4 
CLIP_VALUE = 0.1
NUM_ENV = 6
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5

DELTA_INTERVAL = 0.05
IMPROVED = True
IF_TRAIN = True

EventsWriter = SummaryWriter()

TOTAL_REWARD = 0
BUFFER_REWARD = []      # 存近几次reward
BUFFER_REWARD_LEN = 20  # 缓存reward的episode的长度 
SUCCESS_FLAG_BUFFER = []      # 存近几次成功标志

SUCCESS_FLAG = []     # 存所有的成功标志
COLLISION_FLAG = []   # 存所有的碰撞标志

def run(comm, env:Environment, policy, policy_path, action_bound, optimizer):
    buff = []
    global_update = 0 # 全局更新次数，中断后为保持曲线连续性需更改
    global_step = 0

    if env.index == 0:
        env.reset_world()

    id = 0   # 全局EPISODE，中断后为保持曲线连续性需更改
    while id < MAX_EPISODES:

        env.reset_pose()
        env.generate_goal_point()
        terminal = False
        ep_reward = 0
        step = 1

        laser_obs = env.get_laser_observation()
        laser_stack = deque([laser_obs, laser_obs, laser_obs])
        camera_obs = env.get_camera_observation()
        obs_predict = env.get_observation_predict()
        
        goal = np.asarray(env.get_local_goal())     # 目标相对于小车的角度
        speed = np.asarray(env.get_self_speed())    # 线速度和角速度
        orientation = np.asarray(env.get_local_orientation())

        if IMPROVED:
            state = [laser_stack, camera_obs, obs_predict, goal, speed, orientation]
        else:
            state = [laser_stack, goal, speed]

        # 分布式收集数据
        while not terminal and not rospy.is_shutdown():
            # 收集所有进程的观测空间
            state_list = comm.gather(state, root=0)

            # generate actions at rank==0
            if IMPROVED:
                v, a, logprob, scaled_action = generate_action_diffusion(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)
            else:
                v, a, logprob, scaled_action = generate_action(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)

            # master分发全局策略生成的动作
            real_action = comm.scatter(scaled_action, root=0)
            # 分布式执行
            env.control_vel(real_action)

            # rate.sleep()
            rospy.sleep(DELTA_INTERVAL)

            # get informtion
            r, terminal, result = env.get_reward_and_terminate1(step)

            ep_reward += r
            global_step += 1


            # get next state
            s_next = env.get_laser_observation()
            laser_stack.popleft()
            laser_stack.append(s_next)

            img_next = env.get_camera_observation()
            pre_next = env.get_observation_predict()
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            orientation_next = np.asarray(env.get_local_orientation())
            if IMPROVED:
                state_next = [laser_stack, img_next, pre_next, goal_next, speed_next, orientation_next]
            else:
                state_next = [laser_stack, img_next, goal_next, speed_next]
            
            # 全局步数达到设定的界限就把所有rank的数据发给master，生成一个全局的动作
            if global_step % HORIZON == 0:
                state_next_list = comm.gather(state_next, root=0)
                if IMPROVED:
                    last_v, _, _, _ = generate_action_diffusion(env=env, state_list=state_next_list, policy=policy,
                                                               action_bound=action_bound)
                else:
                    last_v, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy,
                                                               action_bound=action_bound)

            # add transitons in buff and update policy
            # 收集所有进程的reward
            r_list = comm.gather(r, root=0)
            # 收集所有进程的结束状态
            terminal_list = comm.gather(terminal, root=0)

            if env.index == 0 and IF_TRAIN:
                # master收集所有的数据进入buff
                # state_list包含state,goal,speed
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                if len(buff) > HORIZON - 1:
                    if IMPROVED:
                        # state, goal, speed, action, reward, done, logprob, value
                        s_batch, img_batch, pre_batch, goal_batch, speed_batch, orientation_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                            transform_buffer_diffusion(buff=buff)
                        # target_batch, advantages_batch
                        t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                        torch.cuda.empty_cache()
                        memory = (s_batch, img_batch, pre_batch, goal_batch, speed_batch, orientation_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)

                        actor_loss, critic_loss, total_loss = ppo_update_stage1_diffusion(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE,
                                                                                         memory=memory, epoch=EPOCH, coeff_entropy=COEFF_ENTROPY,
                                                                                         clip_value=CLIP_VALUE, num_step=HORIZON, num_env=NUM_ENV,
                                                                                         frames=LASER_HIST, obs_size=OBS_SIZE, act_size=ACT_SIZE)
                    else:
                        # state, goal, speed, action, reward, done, logprob, value
                        s_batch, img_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                            transform_buffer(buff=buff)
                        # target_batch, advantages_batch
                        t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                        memory = (s_batch, img_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    
                        actor_loss, critic_loss, total_loss = ppo_update_stage1(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            num_env=NUM_ENV, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)
                    buff = []
                    global_update += 1
                    EventsWriter.add_scalar("actor_loss", actor_loss, global_update)
                    EventsWriter.add_scalar("critic_loss", critic_loss, global_update)
                    EventsWriter.add_scalar("total_loss", total_loss, global_update)
                    print("Global Update Times: "+str(global_update))

            step += 1
            state = state_next


        if env.index == 0:
            if global_update != 0 and global_update % 50 == 0:
                torch.save(policy.state_dict(), policy_path + '/Stage1_{}'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, step %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, distance, result))
        logger_cal.info(ep_reward)
        
        if env.index == 0:
            global TOTAL_REWARD, BUFFER_REWARD, BUFFER_REWARD_LEN
            TOTAL_REWARD += ep_reward
            EventsWriter.add_scalar("avg_reward", TOTAL_REWARD/(id+1), id+1)

            if len(BUFFER_REWARD) >= BUFFER_REWARD_LEN:
                BUFFER_REWARD.pop(0)
            BUFFER_REWARD.append(ep_reward)

            if len(SUCCESS_FLAG_BUFFER) >= BUFFER_REWARD_LEN:
                SUCCESS_FLAG_BUFFER.pop(0)
            if result == "Reach Goal":
                SUCCESS_FLAG_BUFFER.append(1)
                SUCCESS_FLAG.append(1)
            elif result == "Crashed":
                SUCCESS_FLAG_BUFFER.append(2)
                COLLISION_FLAG.append(1)
            elif result == "Rollover":
                SUCCESS_FLAG_BUFFER.append(3)
            elif result == "Time out":
                SUCCESS_FLAG_BUFFER.append(4)
            else:
                SUCCESS_FLAG_BUFFER.append(0)

            print("id, success_rate, collision_rate: ", id + 1, len(SUCCESS_FLAG) / (id + 1), len(COLLISION_FLAG) / (id + 1))
            EventsWriter.add_scalar("avg_success_rate", len(SUCCESS_FLAG)/(id + 1), id+1)
            EventsWriter.add_scalar("avg_collison_rate", len(COLLISION_FLAG)/(id + 1), id+1)

            EventsWriter.add_scalar("avg_success_rate_buffer", SUCCESS_FLAG_BUFFER.count(1)/BUFFER_REWARD_LEN, id+1)
            EventsWriter.add_scalar("avg_collison_rate_buffer", SUCCESS_FLAG_BUFFER.count(2)/BUFFER_REWARD_LEN, id+1)

            MAX_REWARD = max(BUFFER_REWARD)
            EventsWriter.add_scalar("max_reward_buffer", MAX_REWARD, id+1)
            EventsWriter.add_scalar("avg_reward_buffer", sum(BUFFER_REWARD)/BUFFER_REWARD_LEN, id+1)

        id += 1


if __name__ == '__main__':
    # config log
    hostname = socket.gethostname()
    now = datetime.datetime.now()
    dirname = now.strftime("Stage1_%b%d_%H%M")
    dirname = './log/'+dirname
    
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    output_file = dirname + '/output.log'
    cal_file = dirname + '/cal.log'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    env = Environment(LASER_BEAM, index=rank, num_env=NUM_ENV)
    reward = None
    action_bound = [[0, -1], [1, 1]]    # 速度：（0，1） 转向：（-1，1）

    if rank == 0:
        policy_path = 'policy/stage1'
        policy = None

        if IMPROVED :
            print("Use DetourPolicy")
            policy = DetourPolicy(frames=LASER_HIST, action_space=2)
        else :
            print("test")

        policy = policy.cuda()
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/Stage1_'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            if IF_TRAIN:
                logger.info('############Start Training###########')
            else:
                logger.info('############Start Testing###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict, strict=False)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        policy = None
        policy_path = None
        opt = None

    try:
        run(comm=comm, env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass