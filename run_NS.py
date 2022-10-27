#!~miniconda3/envs/pytorch/bin python
# from __future__ import print_function

import numpy as np
import Src.Utils.utils as utils
from Src.NS_parser import Parser
from Src.config import Config
from time import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import logging
import pickle
from Src.Algorithms import NS_utils
import torch
import seaborn as sns
from os import path
from scipy import stats


def animate(n, *fargs):
    trajectories = fargs[0]
    line = fargs[1]
    line.set_xdata(trajectories[:n, 0])
    line.set_ydata(trajectories[:n, 1])
    return line

class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env
        self.state_dim = np.shape(self.env.reset())[0]

        if len(self.env.action_space.shape) > 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))

        self.model = config.algo(config=config)


    def train(self):
        # Learn the model on the environment
        return_history = []
        action_dict = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
        reached_destination = []
        reached_count = 0
        discounted_return_history = np.zeros(self.config.max_episodes)
        episode_length = np.zeros(self.config.max_episodes)
        success_vector = np.zeros(self.config.max_episodes)
        prowls_prediction = np.zeros(self.config.max_episodes)


        logging.basicConfig(filename='log_trial.log', encoding='utf-8', level=logging.DEBUG)
        logging.debug('START TRIAL \n')

        E = self.config.max_episodes
        H = self.env.max_horizon
        #actions = np.empty((self.config.max_episodes, self.env.max_horizon), dtype=object)
        #action_probs = np.zeros((self.config.max_episodes, self.env.max_horizon, 4))
        #states = np.zeros((self.config.max_episodes, self.env.max_horizon+1, 2))
        #rewards = np.zeros((self.config.max_episodes, self.env.max_horizon))
        #target_pos = np.zeros((self.config.max_episodes, self.env.max_horizon, 4))


        ckpt = self.config.save_after
        rm_history, regret, rm, start_ep = [], 0, 0, 0

        windDir = "right"
        targetPos = [0, 0]

        steps = 0
        t0 = time()
        for episode in range(start_ep, self.config.max_episodes):
            logging.debug("EPISODE {}\n\n".format(episode))

            # RESET ENV AND MODEL
            state = self.env.reset()

            if self.config.env_name == 'NS_Wind':
                if windDir != self.env.direction or episode == self.config.max_episodes-1:
                    with open("{}_Model_wind_{}_seed_{}".format(self.config.algo_name, windDir, self.config.seed),
                              "wb") as file_:
                        pickle.dump(self.model, file_, -1)
                    windDir = self.env.direction
            elif self.config.env_name == 'NS_Target':
                if targetPos != self.env.target or episode == self.config.max_episodes-1:
                    with open("{}_Model_Target_{}_seed_{}".format(self.config.algo_name, self.fromTargetToStr(targetPos), self.config.seed),
                              "wb") as file_:
                        pickle.dump(self.model, file_, -1)
                    targetPos = self.env.target
                    #newActor, _, _ = NS_utils.get_Policy(state_dim=2, config=self.config)
                    #self.model.modules[0] = ('actor', newActor)

            self.model.reset()

            logging.debug("Initial State: {}".format(state))

            step, total_r = 0, 0
            discount = 1
            gamma_t = self.config.gamma
            done = False
            trajectories = np.zeros((self.config.max_steps, 2))
            plotTraj = False
            while not done:
                # self.env.render(mode='human')
                logging.debug("Step: {} \n".format(step))

                if plotTraj:
                    trajectories[step, :] = state

                # GET ACTION
                action, extra_info, dist = self.model.get_action(state)

                #states[episode, step] = self.env.curr_pos
                #actions[episode, step] = action_dict[action]
                #action_probs[episode, step] = dist

                logging.debug("State: {}".format(self.env.curr_pos))
                logging.debug("Action chosen: {}".format(action_dict[action]))
                logging.debug("Actions probabilities (up, right, down, left): {}".format(dist))

                # STEP
                new_state, reward, done, info = self.env.step(action=action)

                if reward > 0:
                    success_vector[episode] = 1

                logging.debug("Went to: {}".format(self.env.curr_pos))
                logging.debug("Reward: {}".format(reward))

                if reward == 29.5:
                    logging.debug("DESTINATION REACHED IN {} STEPS".format(step))

                logging.debug("\n\n\n")

                #rewards[episode, step] = reward # this is not discounted
                #target_pos[episode, step] = self.env.G1
                #if done:
                    #states[episode, step+1] = self.env.curr_pos

                # UPDATE MODEL
                self.model.update(state, action, extra_info, reward, new_state, done)
                #prowls_prediction[episode:episode + self.config.delta] = self.model.prediction
                state = new_state

                # Tracking intra-episode progress
                total_r += reward
                discounted_return_history[episode] += reward * discount
                discount *= gamma_t
                # regret += (reward - info['Max'])
                step += 1
                if step >= self.config.max_steps:
                    break

            if plotTraj:
                fig = plt.figure()
                ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
                line, = ax.plot([], [], lw=2)
                ep_traj = np.reshape(trajectories[np.logical_and.reduce(trajectories[:, :] != [0, 0], axis=1), :],
                                     (-1, 2))
                # print(ep_traj.shape)
                # print(ep_traj)

                target = plt.Rectangle((0.45, 0.9), 0.1, 0.1, fc='red')
                ax.add_patch(target)
                anim = FuncAnimation(fig, animate, interval=200, fargs=(ep_traj, line))
                plt.show(block=False)
                plt.pause(3)
                plt.close()

            # track inter-episode progress
            episode_length[episode] = step
            steps += step
            rm += total_r
            '''if total_r > 0 and episode >= 0.9 * self.config.max_episodes:
                reached_destination.append(True)
                reached_count += 1
            elif total_r <= 0 and episode >= 0.9 * self.config.max_episodes:
                reached_destination.append(False)'''

            if episode%ckpt == 0 or episode == self.config.max_episodes-1:
                rm_history.append(rm)
                return_history.append(total_r)

                print("{} :: Rewards {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
                      format(episode, rm, steps/ckpt, (time() - t0)/ckpt, (time() - t0)/steps, self.model.entropy, self.model.get_grads()))

                # self.model.save()
                #utils.save_plots(return_history, config=self.config, name='{}_rewards'.format(self.config.seed))

                t0 = time()
                steps = 0

            # every 100 episodes set exploration to 0 and plot 10 trajectories
            trajectories = np.zeros((10, self.config.max_steps, 2))
            if False:
            #if episode % 100 == 0 and episode != 0 or episode == self.config.max_episodes-1:

                print("Project a few episodes without exploration epsilon.")
                #self.env.randomness = 0
                #self.model.state_features.eval()
                #self.model.actor.eval()

                # layers_before = [x.data for x in self.model.state_features.net.parameters()]
                #layers_before = [x.data for x in self.model.actor.fc1.parameters()]
                # print("Before: {}".format(layers_before))

                for i in range(2):
                    state = self.env.reset()
                    self.env.episode -= 1
                    self.env.curr_pos = np.array([np.random.randint(2,9) / 10, 0.5])
                    self.env.curr_state = self.env.make_state()
                    state = self.env.make_state()

                    #self.model.reset()

                    step_test = 0
                    done = False
                    while not done:
                        trajectories[i,step_test,:] = state

                        action, extra_info, dist = self.model.get_action(state)

                        action = np.argmax(dist)
                        #print(action)
                        # STEP
                        new_state, reward, done, info = self.env.step(action=action)
                        # UPDATE MODEL
                        # self.model.update(state, action, extra_info, reward, new_state, done)
                        state = new_state
                        step_test += 1
                        #print(trajectories[i,:,:])

                        if step_test >= self.config.max_steps:
                            break

                    fig = plt.figure()
                    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
                    line, = ax.plot([], [], lw=2)
                    ep_traj = np.reshape(trajectories[i,np.logical_and.reduce(trajectories[i,:,:] != [0,0], axis =1), :], (-1,2))
                    #print(ep_traj.shape)
                    #print(ep_traj)

                    target = plt.Rectangle((0.475, 0.925), 0.05, 0.05, fc='red')
                    ax.add_patch(target)
                    anim = FuncAnimation(fig, animate, interval=200, fargs=(ep_traj, line))
                    plt.show(block=False)
                    plt.pause(3)
                    plt.close()

                #self.env.randomness = 0.25
                #self.model.state_features.train()
                #self.model.actor.train()

                #layers_after = [x.data for x in self.model.state_features.net.parameters()]
                layers_after = [x.data for x in self.model.actor.fc1.parameters()]
                #print("After: {}".format(layers_after))
                #print(torch.equal(layers_before[0], layers_after[0]))
                #print(torch.equal(layers_before[1], layers_after[1]))
                print("Finished plotting.")

        prediction_error = np.abs(discounted_return_history - prowls_prediction)
        #np.save("prediction_error_alg_{}_speed_{}_seed_{}".format(self.config.algo_name,
        #                                                                            self.config.speed,
        #                                                                            self.config.seed),
        #                                                                            prediction_error)
        #np.save("predicted_return_alg_{}_speed_{}_seed_{}".format(self.config.algo_name,
        #                                                                            self.config.speed,
        #                                                                            self.config.seed),
        #                                                                            prowls_prediction)
        np.save("disc_return_history_alg_{}_speed_{}_seed_{}".format(self.config.algo_name,
                                                                              self.config.speed,
                                                                              self.config.seed,), discounted_return_history)

        np.save("episode_length_alg_{}_speed_{}_seed_{}".format(self.config.algo_name, self.config.speed, self.config.seed),
                                                                episode_length)

        np.save("episode_success_alg_{}_speed_{}_seed_{}".format(self.config.algo_name, self.config.speed, self.config.seed),
            success_vector)

        #success_rate = reached_count / len(reached_destination)

        #print("Success rate: {}".format(success_rate))

        ## convert your array into a dataframe
        #heatmap = pd.DataFrame(self.env.heatmap)
        '''actions = pd.DataFrame(actions)
        states = pd.DataFrame(np.reshape(states, (E*(H+1),-1)))
        action_probs = pd.DataFrame(np.reshape(action_probs, (E*H,-1)))
        rewards = pd.DataFrame(rewards)
        target_pos = pd.DataFrame(np.reshape(target_pos, (E*H,-1)))'''

        ## save to xlsx file

        '''actions_filepath = 'actions_{}.xlsx'.format(self.config.max_episodes)
        states_filepath = 'states_{}.xlsx'.format(self.config.max_episodes)
        action_probs_filepath = 'action_probs_{}.xlsx'.format(self.config.max_episodes)
        rewards_filepath = 'rewards_{}.xlsx'.format(self.config.max_episodes)
        target_pos_filepath = 'target_pos_{}.xlsx'.format(self.config.max_episodes)'''
        #heat_path = 'heatmap.xlsx'

        #heatmap.to_excel(heat_path, index=False)
        '''actions.to_excel(actions_filepath, index=False)
        states.to_excel(states_filepath, index=False)
        action_probs.to_excel(action_probs_filepath, index=False)
        rewards.to_excel(rewards_filepath, index=False)
        target_pos.to_excel(target_pos_filepath, index=False)'''

        # plot action map
        '''        
        x = np.arange(0.5,0,-0.05)
        x2 = np.arange(0.5,1,0.05)
        x = np.sort(np.concatenate(([0], x, x2[1:], [1])))
        grid_scale = len(x)

        arrow_positions_x = np.zeros((grid_scale, grid_scale))
        arrow_positions_y = np.zeros((grid_scale, grid_scale))
        arrow_weight = np.zeros((grid_scale, grid_scale))

        arrow_dir_x = np.zeros((grid_scale, grid_scale))
        arrow_dir_y = np.zeros((grid_scale, grid_scale))
        grid_step = 0.05

        # put the position
        for i in range(grid_scale-1):
            for j in range(grid_scale-1):
                arrow_positions_x[i, j] = x[i]
                arrow_positions_y[i,j] = x[j]
                action, prob, _ = self.model.get_action(np.array([arrow_positions_x[i, j],arrow_positions_y[i,j]]))
                arrow_weight[i,j] = prob
                # up
                if action == 0:
                    arrow_dir_x[i, j] = 0
                    arrow_dir_y[i, j] = 1
                # right
                if action == 1:
                    arrow_dir_x[i, j] = 1
                    arrow_dir_y[i, j] = 0
                # down
                if action == 2:
                    arrow_dir_x[i, j] = 0
                    arrow_dir_y[i, j] = -1
                #left
                if action == 3:
                    arrow_dir_x[i, j] = -1
                    arrow_dir_y[i, j] = 0

        fig, ax = plt.subplots(figsize=(grid_scale, grid_scale))
        target = plt.Rectangle((0.45, 0.9), 0.1, 0.1, fc='red')
        ax.quiver(arrow_positions_x, arrow_positions_y, arrow_dir_x, arrow_dir_y, pivot='mid', scale=100, linewidths=arrow_weight.flatten()*0.01, edgecolors='k')
        ax.add_patch(target)
        #plt.show()
        plt.savefig('ActionProbs_{}_seed_{}_algo_{}.png'.format(self.config.max_episodes, self.config.seed, self.config.algo_name))'''



        '''if self.config.debug and self.config.env_name == 'NS_Reco':

            fig1, fig2 = plt.figure(figsize=(8, 6)), plt.figure(figsize=(8, 6))
            ax1, ax2 = fig1.add_subplot(1, 1, 1), fig2.add_subplot(1, 1, 1)

            action_prob = np.array(action_prob).T
            true_rewards = np.array(true_rewards).T

            for idx in range(len(dist)):
                ax1.plot(action_prob[idx])
                ax2.plot(true_rewards[idx])

            plt.show()'''

    '''def train_multiple(self):
        # Learn the model on the environment
        return_array = np.zeros((1,self.config.max_episodes))
        true_rewards = []
        action_prob = []
        return_history = []

        ckpt = self.config.save_after
        rm_history, regret, rm, start_ep = [], 0, 0, 0
        # if self.config.restore:
        #     returns = list(np.load(self.config.paths['results']+"rewards.npy"))
        #     rm = returns[-1]
        #     start_ep = np.size(returns)
        #     print(start_ep)

        steps = 0
        t0 = time()
        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode

            state = self.env.reset()
            self.model.reset()

            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')
                action, extra_info, dist = self.model.get_action(state)
                new_state, reward, done, info = self.env.step(action=action)
                self.model.update(state, action, extra_info, reward, new_state, done)
                state = new_state

                # Tracking intra-episode progress
                total_r += reward
                # regret += (reward - info['Max'])
                step += 1
                if step >= self.config.max_steps:
                    break

            # track inter-episode progress
            # returns.append(total_r)
            steps += step
            # rm = 0.9*rm + 0.1*total_r
            rm += total_r
            return_array[0, episode] = total_r
            if episode%ckpt == 0 or episode == self.config.max_episodes-1:
                rm_history.append(rm)
                return_history.append(total_r)
                if self.config.debug and self.config.env_name == 'NS_Reco':
                    action_prob.append(dist)
                    true_rewards.append(self.env.get_rewards())

                print("{} :: Rewards {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
                      format(episode, rm, steps/ckpt, (time() - t0)/ckpt, (time() - t0)/steps, self.model.entropy, self.model.get_grads()))

                # self.model.save()
                utils.save_plots(return_history, config=self.config, name='{}_rewards'.format(self.config.seed))

                t0 = time()
                steps = 0
        return return_array'''

    def fromTargetToStr(self, target):
        if target == [0, 0]:
            return "TopLeft"
        elif target == [0, 4]:
            return "TopRight"
        elif target == [4, 4]:
            return "BotRight"
        elif target == [4, 0]:
            return "BotLeft"


# @profile
def main(train=True, inc=-1, hyper='default', base=-1):
    t = time()
    args = Parser().get_parser().parse_args()
    save = True

    # Use only on-policy method for oracle
    if args.oracle >= 0:
            args.algo_name = 'ONPG'

    if inc >= 0 and hyper != 'default' and base >= 0:
        args.inc = inc
        args.hyper = hyper
        args.base = base


    n_seeds = 30
    # Training mode
    if train:
        for i in range(n_seeds):
            args.seed = i
            config = Config(args)
            solver = Solver(config=config)
            solver.train()

    print("Total time taken: {}".format(time() - t))

    '''   
    if (save):
        with open("{}_Model_max_ep_{}_seed_{}".format(args.algo_name, args.max_episodes, args.seed), "wb") as file_:
            pickle.dump(solver, file_, -1)'''



if __name__ == "__main__":
        main(train=True)

