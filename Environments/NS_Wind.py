from __future__ import print_function
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.ticker import NullLocator
from Src.Utils.utils import Space
import seaborn as sns


class NS_Wind(object):
    def __init__(self,
                 action_type='discrete',  # 'discrete' {0,1} or 'continuous' [0,1]
                 n_actions=4,
                 oracle=-1,
                 speed=4,
                 debug=True,
                 max_step_length=0.1,
                 max_steps=3):

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

        self.max_horizon = int(max_steps / max_step_length) # this is 30 right now
        self.step_reward = -0.5
        self.collision_reward = 0  # -0.05
        self.movement_reward = 0  # 1
        self.reached_reward = 30
        self.target = [0.45, 0.9, 0.55, 1.0] # target is an area of a square centered in 0,95 with side of 0.1
        self.targetPos = [0.5, 0.95]

        # Wind
        self.wind = np.zeros(2)
        self.direction = None
        #self.wind = [self.step_unit/6, 0]

        #self.reset()

    def changeWind(self):

        if self.direction is None:
            self.direction = "right"
        elif self.direction == "right":
            self.direction = "down"
        elif self.direction == "down":
            self.direction = "left"
        elif self.direction == "left":
            self.direction = "up"
        else:
            self.direction = "right"

        if self.direction == "up":
            self.wind = [0, self.step_unit/2]
        elif self.direction == "down":
            self.wind = [0, -self.step_unit/2]
        elif self.direction == "left":
            self.wind = [-self.step_unit/2, 0]
        elif self.direction == "right":
            self.wind = [self.step_unit/2, 0]

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

        if self.episode % 2000 == 0:
            self.changeWind()

        self.episode += 1
        self.steps_taken = 0

        # Non stationarity with wind changes

        #self.curr_pos = np.array([0.5, 0.5])
        self.curr_pos = np.array([np.random.choice(np.linspace(0.1,1,9,endpoint=False)), 0.5])
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
        delta = motion * self.step_unit + self.wind
        new_pos = self.curr_pos + delta  # Take a unit step in the direction of chosen action

        if self.valid_pos(new_pos):
            #dist = np.linalg.norm(delta)
            #reward += self.movement_reward * dist  # small reward for moving, this does nothing right now
            self.curr_pos = new_pos
            reward += self.get_goal_rewards(self.curr_pos)  #check if the current pos is in target region
            # reward += self.open_gate_condition(self.curr_pos)
        else:
            reward += self.collision_reward

        reward -= np.linalg.norm(self.curr_pos - self.targetPos)
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
    env = NS_Wind(debug=True, n_actions=4)
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

    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))