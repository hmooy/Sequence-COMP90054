import math
import random
from abc import ABC, abstractmethod


class MAB(ABC):
    """
     Class represents a multi-armed bandit
    """
    @abstractmethod
    def FindBestChildNode(self, node, q_func):
        pass


class EpsGreedy(MAB):
    def __init__(self, epsilon):
        """
        @params:
        epsilon = explore probability
        """
        self.epsilon = epsilon

    def FindBestChildNode(self, node, q_func):
        """
        @params:
        node = expanded node
        q_func = the q value expression

        Find the best node using Epsilon Greedy
        """
        if random.random() > self.epsilon:
            # if rand() > epsilon, exploit
            max_q_value = float('-inf')
            best_child = None
            for child in node.children:
                if child.state.N == 0:
                    return child
                else:
                    tmp = q_func(child)
                    if tmp > max_q_value:
                        max_q_value = tmp
                        best_child = child
            return best_child
        else:
            # if rand() < epsilon, explore
            return random.choice(node.children)


class UCB(MAB):
    def __init__(self, exploration_constant):
        """
        @params:
        exploration_constant = exploration_constant for UCB, explore-exploit trade-off.
        """
        self.exploration_constant = exploration_constant

    def FindBestChildNode(self, node, q_func):
        """
        @params:
        node = expanded node
        q_func = the q value expression

        Find the best node using UCB, consider the nodes haven't been expanded first.
        """
        parent_N = node.state.N
        max_ucb = float('-inf')
        best_child = None
        for child in node.children:
            if child.state.N == 0:
                return child
            else:
                tmp = self._CalculateUCB(parent_N, q_func(child), child.state.N)
                if tmp > max_ucb:
                    max_ucb = tmp
                    best_child = child
        return best_child

    def _CalculateUCB(self, total_visit, q_value, node_N):
        """Calculate UCB value """
        if node_N == 0:
            return float('inf')
        return q_value + self.exploration_constant * math.sqrt(math.log(total_visit) / node_N)

