from Src.Utils.Policy import Policy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch import tensor, float32
from torch.distributions import Normal
import math

class Clamp(torch.autograd.Function):
    """
    Clamp class with derivative always equal to 1

    --Example
    x = torch.tensor([1,2,3])
    my = Clamp()
    y = my.apply(x, min=2,max=3)
    """
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def get_Policy(state_dim, config):
    if config.cont_actions:
        atype = torch.float32
        actor = Insulin_Gaussian(state_dim=state_dim, config=config)
        action_size = actor.action_dim
    elif config.algo_name == "ContONPG":
        atype = torch.long
        action_size = 1
        actor = ContCategorical(state_dim=state_dim, config=config)
    else:
        atype = torch.long
        action_size = 1
        actor = Categorical(state_dim=state_dim, config=config)

    return actor, atype, action_size

class ContCategorical(Policy):
    def __init__(self, state_dim, config, action_dim=None):
        super(ContCategorical, self).__init__(state_dim, config)

        # overrides the action dim variable defined by super-class
        if action_dim is not None:
            self.action_dim = action_dim

        hiddenLayerDim = 16
        self.l1 = nn.Linear(self.state_dim, hiddenLayerDim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hiddenLayerDim, self.action_dim)


        # Initialise the weights
        torch.nn.init.kaiming_uniform_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.l2.weight)
        torch.nn.init.zeros_(self.l1.bias)
        torch.nn.init.zeros_(self.l2.bias)

        # Continual Backprop parameters
        self.hiddenUnits = np.zeros((hiddenLayerDim))
        self.hiddenUnitsAvg = np.zeros((hiddenLayerDim))
        self.hiddenUnitsAvgBias = np.zeros((hiddenLayerDim))
        self.hiddenUnitsAge = np.zeros((hiddenLayerDim))
        self.hiddenUnitsCount = np.zeros((hiddenLayerDim))
        self.hiddenUtilityBias = np.zeros((hiddenLayerDim))
        self.hiddenUtility = np.zeros((hiddenLayerDim))
        self.nHiddenLayers = 1

        self.init()
        self.replacementRate = 1e-4
        self.decayRate = 0.99
        self.maturityThreshold = 100
        self.unitsToReplace = 0
        self.activation = {}

    def re_init_optim(self):
        self.optim = self.config.optim(self.parameters(), lr=self.config.actor_lr)

    def getActivation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def forward(self, x, saveFeatures=True):
        hook1 = self.a1.register_forward_hook(self.getActivation('h1'))
        #print(x.dtype)
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        hook1.remove()

        if not saveFeatures:
            return x

        # Update count
        self.hiddenUnitsCount += 1
        # Update hidden units estimates
        # Take hidden units values from dictionary
        self.hiddenUnits = np.reshape(self.activation['h1'].detach().numpy(),
                                            (self.hiddenUnitsAvgBias.shape[0]))

        # Unbiased estimate. Warning: uses old mean estimate of the hidden units.
        self.hiddenUnitsAvg = self.hiddenUnitsAvgBias / (1 - np.power(self.decayRate, self.hiddenUnitsCount))
        if not np.isfinite(self.hiddenUnitsAvg).all():
            print("There are inf values in hiddenUnitsAvg")
        # Biased estimate: updated with current hidden units values
        self.hiddenUnitsAvgBias = self.decayRate * self.hiddenUnitsAvgBias + \
                                  (1 - self.decayRate) * self.hiddenUnits

        # Compute mean-corrected contribution utility (called z in the paper) actually don't do that

        # Weights going out from layer l to layer l+1.
        # The i-th column of the matrix has the weights of the i-th element of l-th layer.
        # For each layer we have a mxn matrix with n=#units of l-th layer and m=#units in l+i-th layer
        outgoingWeightsH1 = self.state_dict()['l2.weight'].detach().numpy()
        # Sum together contributions (sum elements of same columns) from same hidden unit
        # and reshape to obtain h1xh2 matrix to use in the formula.
        outgoingWeights = np.sum(np.abs(outgoingWeightsH1), axis=0).flatten()

        contribUtility = np.multiply(np.abs(self.hiddenUnits - self.hiddenUnitsAvg), outgoingWeights)

        # Compute the adaptation utility
        # Weights going in from layer l-1 to layer l.
        # The j-th row of the matrix has the weights going in the j-th element of l-th layer.
        # For each layer we have a mxn matrix with n=#units of l-th layer and m=#units in l-1-th layer
        inputWeightsH1 = self.state_dict()['l1.weight'].detach().numpy()
        # Sum together contributions (sum elements of same rows) from same hidden unit
        # and reshape to obtain h1xh2 matrix to use in the formula.
        # The adaptation utility is the element-wise inverse of the inputWeight matrix.
        inputWeights = np.sum(np.abs(inputWeightsH1), axis=1).flatten()

        # Compute hidden unit utility
        self.hiddenUtility = self.hiddenUtilityBias / (1 - np.power(self.decayRate, self.hiddenUnitsCount))
        # Now update the hidden utility with new values
        self.hiddenUtilityBias = self.decayRate * self.hiddenUtilityBias + \
                                 (1 - self.decayRate) * contribUtility / inputWeights

        return x


    def generateAndTest(self):

        # Update hidden units age
        self.hiddenUnitsAge += 1
        nUnits = self.hiddenUnits.shape[0]

        # Select lower utility features depending on the replacement rate
        self.unitsToReplace += self.replacementRate * np.count_nonzero(self.hiddenUnitsAge > self.maturityThreshold)

        while (self.unitsToReplace >= 1):
            # Scan matrix of utilities to find lower element with age > maturityThreshold.
            min = self.hiddenUtility[0]
            minPos = 0
            for i in range(self.hiddenUtility.shape[0]):
                if self.hiddenUtility[i] < min and self.hiddenUnitsAge[i] > self.maturityThreshold:
                    min = self.hiddenUtility[i]
                    minPos = i

            # If the min is in [0,j] it might be too young to be changed
            if (self.hiddenUnitsAge[minPos]) < self.maturityThreshold:
                break
            # Now out min and minPos values are legitimate and we can replace the input weights and set
            # to zero the outgoing weights for the selected hidden unit.
            # Set to 0 the age of the hidden unit.
            self.hiddenUnitsAge[minPos] = 0
            self.hiddenUnitsCount[minPos] = 0
            # Set to 0 the utilities and mean values of the hidden unit.
            self.hiddenUtilityBias[minPos] = 0
            self.hiddenUnitsAvgBias[minPos] = 0

            # Reset weights

            # Take state_dict
            weights = self.state_dict()
            # Reinitialise input weights (i-th row of previous layer)
            temp = torch.empty((1, weights['l1.weight'].shape[1]))
            torch.nn.init.kaiming_uniform_(temp, mode='fan_in', nonlinearity='relu')
            weights['l1.weight'][minPos, :] = temp
            # Reset the input bias
            weights['l1.bias'][minPos] = 0
            # Set to 0 outgoing weights (i-th column of next layer) and do the same for bias
            weights['l2.weight'][:, minPos] = 0
            #weights['l2.bias'][i] = 0
            # Load stat_dict to the model to save changes
            self.load_state_dict(weights)
            # We replaced a hidden unit, reduce counter.
            self.unitsToReplace -= 1

    def get_action_w_prob_dist(self, state, explore=0):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        #print("Dist: {}".format(dist))

        probs = dist.cpu().view(-1).data.numpy()
        #print("Prob: {}".format(probs))
        action = np.random.choice(self.action_dim, p=probs)

        return action, probs[action], probs

    def get_prob(self, state, action):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        return dist.gather(1, action), dist

    def get_logprob_dist(self, state, action):
        x = self.forward(state, saveFeatures=False)                                                         # BxA
        log_dist = F.log_softmax(x, -1)                                                      # BxA
        return log_dist.gather(1, action), log_dist                                          # BxAx(Bx1) -> B



'''class ContCategorical(Policy):
    def __init__(self, state_dim, config, action_dim=None):
        super(ContCategorical, self).__init__(state_dim, config)

        # overrides the action dim variable defined by super-class
        if action_dim is not None:
            self.action_dim = action_dim

        hiddenLayerDim = 16
        self.l1 = nn.Linear(self.state_dim, hiddenLayerDim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hiddenLayerDim, hiddenLayerDim)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(hiddenLayerDim, self.action_dim)

        # Initialise the weights
        torch.nn.init.kaiming_uniform_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.l2.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.l3.weight, mode='fan_in', nonlinearity='relu')

        # Continual Backprop parameters
        self.hiddenUnits = np.zeros((hiddenLayerDim, 2))
        self.hiddenUnitsAvg = np.zeros((hiddenLayerDim, 2))
        self.hiddenUnitsAvgBias = np.zeros((hiddenLayerDim, 2))
        self.hiddenUnitsAge = np.zeros((hiddenLayerDim, 2))
        self.hiddenUnitsCount = np.zeros((hiddenLayerDim, 2))
        self.hiddenUtilityBias = np.zeros((hiddenLayerDim, 2))
        self.hiddenUtility = np.zeros((hiddenLayerDim, 2))
        self.nHiddenLayers = 2

        self.init()
        self.replacementRate = 0.1
        self.decayRate = 0.99
        self.maturityThreshold = 100
        self.unitsToReplace = 0
        self.activation = {}

    def re_init_optim(self):
        self.optim = self.config.optim(self.parameters(), lr=self.config.actor_lr)

    def getActivation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def forward(self, x, saveFeatures=True):
        hook1 = self.a1.register_forward_hook(self.getActivation('h1'))
        hook2 = self.a2.register_forward_hook(self.getActivation('h2'))
        #print(x.dtype)
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        hook1.remove()
        hook2.remove()

        if not saveFeatures:
            return x

        # Update count
        self.hiddenUnitsCount += 1
        # Update hidden units estimates
        # Take hidden units values from dictionary
        self.hiddenUnits[:, 0] = np.reshape(self.activation['h1'].detach().numpy(),
                                            (self.hiddenUnitsAvgBias.shape[0]))
        self.hiddenUnits[:, 1] = np.reshape(self.activation['h2'].detach().numpy(),
                                            (self.hiddenUnitsAvgBias.shape[0]))

        # Unbiased estimate. Warning: uses old mean estimate of the hidden units.
        self.hiddenUnitsAvg = self.hiddenUnitsAvgBias / (1 - np.power(self.decayRate, self.hiddenUnitsCount))
        if not np.isfinite(self.hiddenUnitsAvg).all():
            print("There are inf values in hiddenUnitsAvg")
        # Biased estimate: updated with current hidden units values
        self.hiddenUnitsAvgBias = self.decayRate * self.hiddenUnitsAvgBias + \
                                  (1 - self.decayRate) * self.hiddenUnits

        # Compute mean-corrected contribution utility (called z in the paper) actually don't do that

        # Weights going out from layer l to layer l+1.
        # The i-th column of the matrix has the weights of the i-th element of l-th layer.
        # For each layer we have a mxn matrix with n=#units of l-th layer and m=#units in l+i-th layer
        outgoingWeightsH1 = self.state_dict()['l2.weight'].detach().numpy()
        outgoingWeightsH2 = self.state_dict()['l3.weight'].detach().numpy()
        # Sum together contributions (sum elements of same columns) from same hidden unit
        # and reshape to obtain h1xh2 matrix to use in the formula.
        outgoingWeights = np.hstack((np.reshape(np.sum(np.abs(outgoingWeightsH1), axis=0), (-1, 1)),
                                     np.reshape(np.sum(np.abs(outgoingWeightsH2), axis=0), (-1, 1))))

        contribUtility = np.multiply(np.abs(self.hiddenUnits - self.hiddenUnitsAvg), outgoingWeights)

        # Compute the adaptation utility
        # Weights going in from layer l-1 to layer l.
        # The j-th row of the matrix has the weights going in the j-th element of l-th layer.
        # For each layer we have a mxn matrix with n=#units of l-th layer and m=#units in l-1-th layer
        inputWeightsH1 = self.state_dict()['l1.weight'].detach().numpy()
        inputWeightsH2 = self.state_dict()['l2.weight'].detach().numpy()
        # Sum together contributions (sum elements of same rows) from same hidden unit
        # and reshape to obtain h1xh2 matrix to use in the formula.
        # The adaptation utility is the element-wise inverse of the inputWeight matrix.
        inputWeights = np.hstack((np.reshape(np.sum(np.abs(inputWeightsH1), axis=1), (-1, 1)),
                                  np.reshape(np.sum(np.abs(inputWeightsH2), axis=1), (-1, 1))))

        # Compute hidden unit utility
        self.hiddenUtility = self.hiddenUtilityBias / (1 - np.power(self.decayRate, self.hiddenUnitsCount))
        # Now update the hidden utility with new values
        self.hiddenUtilityBias = self.decayRate * self.hiddenUtilityBias + \
                                 (1 - self.decayRate) * contribUtility / inputWeights

        return x


    def generateAndTest(self):

        # Update hidden units age
        self.hiddenUnitsAge += 1
        nUnits = self.hiddenUnits.shape[0]

        # Do the same for each layer.
        for j in range(self.hiddenUtility.shape[1]):
            # Select lower utility features depending on the replacement rate
            unitsToReplace = math.ceil(self.replacementRate * np.count_nonzero(self.hiddenUnitsAge[:, j] > self.maturityThreshold))

            while (unitsToReplace > 0):
                # Scan matrix of utilities to find lower element with age > maturityThreshold.
                min = self.hiddenUtility[0, j]
                minPos = 0
                for i in range(self.hiddenUtility.shape[0]):
                    if self.hiddenUtility[i, j] < min and self.hiddenUnitsAge[i, j] > self.maturityThreshold:
                        min = self.hiddenUtility[i, j]
                        minPos = i

                # If the min is in [0,j] it might be too young to be changed
                if (self.hiddenUnitsAge[minPos, j]) < self.maturityThreshold:
                    break
                # Now out min and minPos values are legitimate and we can replace the input weights and set
                # to zero the outgoing weights for the selected hidden unit.
                # Set to 0 the age of the hidden unit.
                self.hiddenUnitsAge[minPos, j] = 0
                self.hiddenUnitsCount[minPos, j] = 0
                # Set to 0 the utilities and mean values of the hidden unit.
                self.hiddenUtilityBias[minPos, j] = 0
                self.hiddenUnitsAvgBias[minPos, j] = 0

                # Reset weights
                # If first hidden layer
                if j == 0:
                    # Take state_dict
                    # TODO: check if the initialisation now is different than the one I do at the beginning (depends on # of units?)
                    weights = self.state_dict()
                    # Reinitialise input weights (i-th row of previous layer)
                    temp = torch.empty((1, weights['l1.weight'].shape[1]))
                    torch.nn.init.kaiming_uniform_(temp, mode='fan_in', nonlinearity='relu')
                    weights['l1.weight'][minPos, :] = temp
                    # Reset the input bias
                    weights['l1.bias'][minPos] = 0
                    # Set to 0 outgoing weights (i-th column of next layer) and do the same for bias
                    weights['l2.weight'][:, minPos] = 0
                    #weights['l2.bias'][i] = 0
                    # Load stat_dict to the model to save changes
                    self.load_state_dict(weights)

                # If second hidden layer.
                if j == 1:
                    # Take state_dict
                    weights = self.state_dict()
                    # Reinitialise input weights (i-th row of previous layer)
                    temp = torch.empty((1, weights['l2.weight'].shape[1]))
                    torch.nn.init.kaiming_uniform_(temp, mode='fan_in', nonlinearity='relu')
                    weights['l2.weight'][minPos, :] = temp
                    # Reset the input bias
                    weights['l2.bias'][minPos] = 0
                    # Set to 0 outgoing weights (i-th column of next layer) and do the same for bias.
                    weights['l3.weight'][:, minPos] = 0
                    #weights['l3.bias'][i] = 0
                    # Load stat_dict to the model to save changes
                    self.load_state_dict(weights)

                # We replaced a hidden unit, reduce counter.
                unitsToReplace -= 1

    def get_action_w_prob_dist(self, state, explore=0):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        #print("Dist: {}".format(dist))

        probs = dist.cpu().view(-1).data.numpy()
        #print("Prob: {}".format(probs))
        action = np.random.choice(self.action_dim, p=probs)

        return action, probs[action], probs

    def get_prob(self, state, action):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        return dist.gather(1, action), dist

    def get_logprob_dist(self, state, action):
        x = self.forward(state, saveFeatures=False)                                                         # BxA
        log_dist = F.log_softmax(x, -1)                                                      # BxA
        return log_dist.gather(1, action), log_dist                                          # BxAx(Bx1) -> B
'''
# Keep the old actor without continualbackprop

class Categorical(Policy):
    def __init__(self, state_dim, config, action_dim=None):
        super(Categorical, self).__init__(state_dim, config)

        # overrides the action dim variable defined by super-class
        if action_dim is not None:
            self.action_dim = action_dim

        hiddenLayerDim = 16
        self.l1 = nn.Linear(self.state_dim, hiddenLayerDim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hiddenLayerDim, self.action_dim)

        # Initialise the weights
        torch.nn.init.kaiming_uniform_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.l2.weight)
        torch.nn.init.zeros_(self.l1.bias)
        torch.nn.init.zeros_(self.l2.bias)

        self.init()

    def re_init_optim(self):
        self.optim = self.config.optim(self.parameters(), lr=self.config.actor_lr)

    def forward(self, x, saveFeatures=True):
        # print(x.dtype)
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)

        return x

    def get_action_w_prob_dist(self, state, explore=0):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        #print("Dist: {}".format(dist))

        probs = dist.cpu().view(-1).data.numpy()
        #print("Prob: {}".format(probs))
        action = np.random.choice(self.action_dim, p=probs)

        return action, probs[action], probs

    def get_prob(self, state, action):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        return dist.gather(1, action), dist

    def get_logprob_dist(self, state, action):
        x = self.forward(state)                                                              # BxA
        log_dist = F.log_softmax(x, -1)                                                      # BxA
        return log_dist.gather(1, action), log_dist                                          # BxAx(Bx1) -> B


class Insulin_Gaussian(Policy):
    def __init__(self, state_dim, config):
        super(Insulin_Gaussian, self).__init__(state_dim, config, action_dim=2)

        # Set the ranges or the actions
        self.low, self.high = config.env.action_space.low * 1.0, config.env.action_space.high * 1.0
        self.action_low = tensor(self.low, dtype=float32, requires_grad=False, device=config.device)
        self.action_diff = tensor(self.high - self.low, dtype=float32, requires_grad=False, device=config.device)

        print("Action Low: {} :: Action High: {}".format(self.low, self.high))

        # Initialize network architecture and optimizer
        self.fc_mean = nn.Linear(state_dim, 2)
        if self.config.gauss_std > 0:
            self.forward = self.forward_wo_var
        else:
            self.fc_var = nn.Linear(state_dim, self.action_dim)
            self.forward = self.forward_with_var
        self.init()

    def forward_wo_var(self, state):
        action_mean = torch.sigmoid(self.fc_mean(state)) * self.action_diff + self.action_low       # BxD -> BxA
        std = torch.ones_like(action_mean, requires_grad=False) * self.config.gauss_std             # BxD -> BxA
        return action_mean, std

    def forward_with_var(self, state):
        action_mean = torch.sigmoid(self.fc_mean(state)) * self.action_diff + self.action_low       # BxD -> BxA
        action_std = torch.sigmoid(self.fc_var(state)) + 1e-2                                       # BxD -> BxA
        return action_mean, action_std

    def get_action_w_prob_dist(self, state, explore=0):
        # Pytorch doesn't have a direct function for computing prob, only log_prob.
        # Hence going the round-about way.
        action, logp, dist = self.get_action_w_logprob_dist(state, explore)
        prob = np.exp(logp)

        return action, prob, dist

    def get_prob(self, state, action):
        logp, dist = self.get_logprob_dist(state, action)
        return torch.exp(logp), dist                                                            # B, BxAx(dist)


    def get_action_w_logprob_dist(self, state, explore=0):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()

        # prob = poduct of all probabilities. Therefore log is the sum of them.
        logp = dist.log_prob(action).view(-1).data.numpy().sum(axis=-1)
        action = action.cpu().view(-1).data.numpy()

        return action, logp, dist

    def get_logprob_dist(self, state, action):
        mean, var = self.forward(state)                                                         # BxA, BxA
        dist = Normal(mean, var)                                                                # BxAxdist()
        return dist.log_prob(action).sum(dim=-1), dist                                          # BxAx(BxA) -> B
