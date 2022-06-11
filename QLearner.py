import random as rand
import numpy as np


class QLearner(object):
    """
    This is a Q learner object.

    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available..
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """

    def __init__(
            self,
            num_states=100,
            num_actions=4,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
    ):
        """
        Constructor method
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        # Q[s,a]
        self.q_table = np.zeros((self.num_states, self.num_actions))
        # Transition and Reward: TR[s,a,s',r]
        self.tr_table = []

    def querysetstate(self, s):
        """
        Update the state without updating the Q-table

        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """

        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[s])

        self.s = s
        self.a = action

        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def query(self, s_prime, r):
        """
        Update the Q table and return an action

        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """

        s = self.s
        a = self.a
        alpha = self.alpha
        gamma = self.gamma
        # update rule/Q-table
        improved_est = r + gamma * self.q_table[s_prime, np.argmax(self.q_table[s_prime])]
        self.q_table[s, a] = (1 - alpha) * self.q_table[s, a] + alpha * improved_est

        # Dyna-Q
        if self.dyna > 0:
            # update model
            self.tr_table.append((s, a, s_prime, r))
            for i in range(self.dyna):
                # hallucinate
                r_idx = rand.randint(0, len(self.tr_table)-1)
                s_d = self.tr_table[r_idx][0]
                a_d = self.tr_table[r_idx][1]
                s_prime_d = self.tr_table[r_idx][2]
                r_d = self.tr_table[r_idx][3]
                # Q update
                improved_est = r_d + gamma * self.q_table[s_prime_d, np.argmax(self.q_table[s_prime_d])]
                self.q_table[s_d, a_d] = (1 - alpha) * self.q_table[s_d, a_d] + alpha * improved_est

        # update state and action
        action = self.querysetstate(s_prime)
        # update random action rate
        self.rar = self.rar * self.radr

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")
        return action


if __name__ == "__main__":
    pass