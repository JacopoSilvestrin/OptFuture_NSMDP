from __future__ import print_function
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.ticker import NullLocator
from Src.Utils.utils import Space
import seaborn as sns
import numpy as np
from Src.Utils.utils import Space
#import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')
AGENT_N = 1
TARGET_N = 2
FLOOR_N = 3

colors = {1: (255,275,0),
          2: (0,255, 0),
          3: (211,211,211)}

possible_actions = {0: "Right",
                    1: "Down",
                    2: "Left",
                    3: "Up"}


class NS_Target(object):

    def __init__(self, deterministic = False, speed=2, oracle=-1, debug=True):
        self.n_rows = 5
        self.n_cols = 5
        #self.actions = ['up', 'down', 'left', 'right']
        self.n_actions = 4
        self.motions = np.zeros((self.n_actions, 2), dtype=int)
        self.motions[0] = [0, 1] # right
        self.motions[1] = [1, 0] # down
        self.motions[2] = [0, -1] # left
        self.motions[3] = [-1, 0] # up

        self.target = [0,0]
        self.collision_reward = -1
        self.step_reward = -0.5
        self.target_reward = 10

        # make one hot encoding of state
        #self.state_combination = self.get_state_comb()

        self.action_space = Space(size=self.n_actions)
        self.observation_space = Space(low=np.zeros(2), high=np.array([self.n_rows-1,self.n_cols-1]))
        self.max_horizon = 25

        self.episode = 0
        self.steps_taken = 0

        self.reset()

    def seed(self, seed):
        self.seeding = seed

    def reset(self):
        """
        Sets the environment to default conditions
        :return: None
        """
        self.episode += 1
        self.steps_taken = 0

        if self.episode % 2000 == 0:
            self.changeTarget()

        self.curr_pos = np.array([2, 2])

        return self.curr_pos

    def changeTarget(self):
        if self.target == [0, 0]:
            self.target = [0, 4]
        elif self.target == [0, 4]:
            self.target = [4, 4]
        elif self.target == [4, 4]:
            self.target = [4, 0]
        elif self.target == [4, 0]:
            self.target = [0, 0]


    def isDone(self):
        if all(self.curr_pos == self.target) or self.steps_taken >= self.max_horizon:
            return True
        else:
            return False


    def step(self, action):
        self.steps_taken += 1
        reward = 0

        # Check if previous state was end of MDP, if it was, then we are in absorbing state currently.
        # Terminal state has a Self-loop and a 0 reward
        done = self.isDone()  # if in area of target or if reached max steps
        if done:
            return self.curr_pos, 0, done, {'No INFO implemented yet'}

        reward += self.step_reward

        new_pos = self.curr_pos + self.motions[action]  # Take a unit step in the direction of chosen action

        if self.validPos(new_pos):
            self.curr_pos = new_pos
            # check if the current pos is in target region
            if self.isDone():
                reward += self.target_reward
                print("Got reward {} in {} steps!! ".format(reward, self.steps_taken))

        else:
            reward += self.collision_reward

        return self.curr_pos.copy(), reward, self.isDone(), {'No INFO implemented yet'}


    def render(self):

        env = np.zeros((self.n_rows, self.n_cols, 3), dtype=np.uint8)
        env[self.curr_pos[0], self.curr_pos[1], :] = colors[AGENT_N]
        env[self.target[0], self.target[1], :] = colors[TARGET_N]
        plt.imshow(env)
        plt.show(block=False)
        #plt.ion()
        plt.pause(0.1)
        #plt.close()


    def validPos(self, curr_pos):
        valid = True
        x, y = curr_pos
        # if went out of grid or against wall reward is -1
        if x < 0 or x >= self.n_rows or y < 0 or y >= self.n_cols:
            valid = False
        return valid


if __name__ == "__main__":
    # Random Agent
    rewards_list = np.empty((0,2))
    env = NS_Grid()
    env.render()

    for i in range(20):
        rewards = np.array([0,0], dtype=float)
        done = [False, False]
        env.reset()
        #count = 0

        while not all(done):
            env.render()
            action_a = np.random.randint(0,4)
            action_b = np.random.randint(0,4)
            action = [action_a, action_b]
            print("Action A: ", possible_actions[action_a])
            print("Action B: ", possible_actions[action_b])
            print("Do Step")
            next_state, r, done, _ = env.step(action)
            print("Next state A: ", next_state[0])
            print("Next State B: ", next_state[1])
            rewards += r
            #count += 1
        rewards_list = np.vstack([rewards_list, rewards])

    #print("Average random rewards: ", np.mean(rewards_list, axis=0), np.sum(rewards_list, axis=0))
'''
class NS_Target(object):
    def __init__(self,
                 action_type='discrete',  # 'discrete' {0,1} or 'continuous' [0,1]
                 n_actions=4,
                 oracle=-1,
                 speed=4,
                 debug=True,
                 max_step_length=0.1,
                 max_steps=4):

        n_actions = 4
        self.debug = debug

        # NS Specific settings
        self.oracle = oracle
        self.speed = speed
        self.episode = 0

        self.n_actions = n_actions
        self.action_space = Space(size=n_actions)
        self.observation_space = Space(low=np.zeros(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)
        self.disp_flag = False

        self.motions = self.get_action_motions(self.n_actions)

        self.step_unit = 0.1

        self.max_horizon = int(max_steps / max_step_length) # this is 40 right now
        self.step_reward = -0.5
        self.collision_reward = -0.5  # -0.05
        self.movement_reward = 0  # 1
        self.reached_reward = 30
        self.target = [0.0, 0.9, 0.1, 1.0] # target is an area in top left corner and moves clockwise
        self.targetCenter = [0.05, 0.95]
        #self.targetPos = [0.5, 0.95]
        #self.target = [0.9, 0.9, 1.0, 1.0]
        #self.targetCenter = [0.95, 0.95]

        #self.reset()

    def changeTarget(self):
        # Move clockwise target
        if self.target == [0.0, 0.9, 0.1, 1.0]:
            self.target = [0.45, 0.9, 0.55, 1.0]
            self.targetCenter = [0.5, 0.95]
        elif self.target == [0.45, 0.9, 0.55, 1.0]:
            self.target = [0.9, 0.9, 1.0, 1.0]
            self.targetCenter = [0.95, 0.95]
        elif self.target == [0.9, 0.9, 1.0, 1.0]:
            self.target = [0.0, 0.9, 0.1, 1.0]
            self.targetCenter = [0.05, 0.95]
        else:
            print("Something is very wrong.")


    def seed(self, seed):
        self.seed = seed

    def render(self):
        x, y = self.curr_pos

        # ----------------- One Time Set-up --------------------------------
        if not self.disp_flag:
            self.disp_flag = True
            # plt.axis('off')
            self.currentAxis = plt.gca()
            plt.figure(1, frameon=False)                            #Turns off the the boundary padding
            self.currentAxis.xaxis.set_major_locator(NullLocator()) #Turns of ticks of x axis
            self.currentAxis.yaxis.set_major_locator(NullLocator()) #Turns of ticks of y axis
            plt.ion()                                               #To avoid display blockage



    def reset(self):
        """
        Sets the environment to default conditions
        :return: None
        """

        self.episode += 1
        self.steps_taken = 0

        # Non stationarity
        if self.episode % 4000 == 0:
            self.changeTarget()

        self.curr_pos = np.array([0.5, 0.5])
        self.curr_state = self.make_state()

        return self.curr_state


    def get_action_motions(self, n_actions):
        shape = (n_actions, 2)
        motions = np.zeros(shape)

        motions[0] = [0, 1]
        motions[1] = [1, 0]
        motions[2] = [0, -1]
        motions[3] = [-1, 0]

        return motions

    def step(self, action):
        # action = binaryEncoding(action, self.n_actions) # Done by look-up table instead
        self.steps_taken += 1
        reward = 0

        # Check if previous state was end of MDP, if it was, then we are in absorbing state currently.
        # Terminal state has a Self-loop and a 0 reward
        term = self.is_terminal() # if in area of target or if reached max steps
        if term:
            return self.curr_state, 0, term, {'No INFO implemented yet'}

        motion = self.motions[action]  # Table look up for the impact/effect of the selected action
        #reward += self.step_reward
        delta = motion * self.step_unit
        new_pos = self.curr_pos + delta  # Take a unit step in the direction of chosen action

        if self.valid_pos(new_pos):
            dist = np.linalg.norm(delta)
            reward += self.movement_reward * dist  # small reward for moving, this does nothing right now
            self.curr_pos = new_pos
            reward += self.get_goal_rewards(self.curr_pos)  #check if the current pos is in target region
            # reward += self.open_gate_condition(self.curr_pos)
        else:
            reward += self.collision_reward

        reward -= np.linalg.norm(self.curr_pos - self.targetCenter)
        # self.update_state()
        self.curr_state = self.make_state()

        return self.curr_state.copy(), reward, self.is_terminal(), {'No INFO implemented yet'}

    def make_state(self):
        x, y = self.curr_pos
        state = [x, y]
        return state

    def get_goal_rewards(self, pos):
        if self.in_region(pos, self.target):
            reward = self.reached_reward
            print("Got reward {} in {} steps!! ".format(reward, self.steps_taken))
            return reward
        return 0

    def valid_pos(self, pos):
        flag = True
        # Check boundary conditions
        if not self.in_region(pos, [0,0,1,1]):
            flag = False
        return flag

    def is_terminal(self):
        if self.in_region(self.curr_pos, self.target):
            return 1
        elif self.steps_taken >= self.max_horizon:
            return 1
        else:
            return 0

    def in_region(self, pos, region):
        x0, y0 = pos
        x1, y1, x2, y2 = region
        if x0 >= x1 and x0 <= x2 and y0 >= y1 and y0 <= y2:
            return True
        else:
            return False


if __name__=="__main__":
    # Random Agent
    rewards_list = []
    env = NS_Target(debug=True, n_actions=4)
    for i in range(1000):
        rewards = 0
        done = False
        env.reset()
        while not done:
            env.render()
            action = np.random.randint(env.n_actions)
            next_state, r, done, _ = env.step(action)
            rewards += r
        rewards_list.append(rewards)

    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))'''