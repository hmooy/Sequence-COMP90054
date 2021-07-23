import time
import copy
import random
from .game_tree import Node, State
from .game_funcs import *
from ..hs_utils.search_problem import SearchProblem
from ..hs_utils.goalstates import GoalState


class MCTS:
    """
    @params:
    player_id = ID of current agent (1,2,3,4)
    game_state = object of SequenceState containing current game state info
    actions = list of all possible actions in dictionary format
    time_limit = limit execution time
    mab = multi-armed bandits value used to select actions
    gamma = gamma value

    This is the MCTS object used by the MCTS player
    """
    def __init__(self, player_id, game_state, actions, time_limit, mab, gamma):
        self.actions = actions
        self.player_id = player_id
        # self.time_limit = time_limit
        self.mab = mab
        self.gamma = gamma
        # Initial tree
        self.root = Node(State(player_id, game_state, None), None)
        # Here we expand the root directly to prevent empty selection
        self.root.ExpandChildren(actions)

    def FindNextMove(self, first_q_func, second_q_func, time_limit):
        """
        @params:
        first_q_func = First Q function which calculates UTB 
        second_q_func = Second Q function 
        time_limit = limit execution time

        This function will return the best action using UTC
        """
        begin_time = time.time()
        while time.time() - begin_time < time_limit:
            # Selection
            expand_node = self.Selection(self.root, first_q_func)
            # Expansion
            child = expand_node
            if expand_node.state.N > 1 and not GameEnds13(expand_node.state.game_state):
                children = self.Expansion(expand_node)
                child = self.Choose(children)
            # Simulation
            rewards, action_count = self.Simulation(child)
            # Back propagation
            self.Back_prop(child, rewards, action_count, self.gamma)
        # Choose the move with max Q value
        best_child = max(self.root.children, key=first_q_func)
        # best_child = self._ChooseBestChildWithTieBreaker(self.root.children, first_q_func, second_q_func)
        return best_child.state.pre_move

    def Selection(self, root, q_func):
        """
        @params:
        root = Root node
        q_func = Q function to calculate UTB

        Return the child node with the highest UCB val
        """
        node = root
        while len(node.children) > 0:
            node = self.mab.FindBestChildNode(node, q_func)
        return node

    @staticmethod
    def Expansion(node):
        node.ExpandChildren()
        return node.children

    @staticmethod
    def Choose(children):
        return random.choice(children)

    @staticmethod
    def Simulation(child):
        """ Use rollout to simulate """
        gs_copy = copy.deepcopy(child.state.game_state)
        current_player_id = child.state.player_id
        action_count = 0

        current_seq_num = sum(gs_copy.agents[i].completed_seqs for i in range(len(gs_copy.agents)))
        target_seq_num = 1 if current_seq_num < 1 else 2
        while not GameEnds13(gs_copy, target_seq_num):
        # while not GameEnds13(gs_copy):
            # Update move
            actions = GetLegalActions13(gs_copy, current_player_id)
            selected = random.choice(actions)
            # selected = MCTS._NaiveMoveSelected(gs_copy, actions,current_player_id)
            gs_copy, current_player_id = Update13(game_state=gs_copy, current_agent_index=current_player_id, action=selected)
            # Change to the opponent
            action_count += 1
        rewards = []
        for i in range(len(gs_copy.agents)):
            rewards.append(gs_copy.agents[i].completed_seqs)
        return rewards, action_count

    @staticmethod
    def Back_prop(node, rewards, action_count, discount_factor):
        """ Backpropagation """
        # Get the rewards after discount
        real_rewards = [reward * (discount_factor ** action_count) for reward in rewards]
        # Back propagation
        p_node = node
        while p_node is not None:
            # Increase the visited count
            p_node.state.N += 1
            # Instead change the win score directly, add it into it.
            for i in range(len(p_node.state.game_state.agents)):
                p_node.state.win_scores_sum[i] += real_rewards[i]
            p_node = p_node.parent
