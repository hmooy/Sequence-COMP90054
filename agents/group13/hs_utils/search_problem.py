from .goalstates import *
from ..game_utils.game_funcs import GenerateSuccessor13,GetLegalActions13

class SearchProblem:
    """
    Class for local search algorithms
    """
    def __init__(self,player_id, goal_states, actions):
        """
        game_state: the current board in list of list format
        goal_state: Object BoardList
        """
        self.actions = actions
        self.player_id = player_id
        self.goal_states = goal_states
    
    def GreedyAlgorithm(self,heuristic = "simple"):
        """
        Greedy heuristic search (local constraint)
        """
        play_cost = []
        draft_cost = []
        play_set = []
        draft_set = []

        try:
            for action in self.actions:
                # print("doing greedy...")
                if (action["play_card"],action["coords"]) in play_set:
                    continue
                play_set.append((action["play_card"],action["coords"]))
                place_h = self.goal_states.PlayAction(action,heuristic)

                if action["draft_card"] in draft_set:
                    continue
                draft_h = self.goal_states.DraftAction(action,heuristic)
                play_cost.append((action,place_h))
                draft_cost.append((action["draft_card"],draft_h))
        except:
            print("Error with Greedy")
        
        best_play_action = sorted(play_cost,key=lambda x: x[1])[0][0]
        best_draft_action = sorted(draft_cost,key=lambda x: x[1])[0][0]
        best_action = best_play_action
        best_action["draft_card"] = best_draft_action
        # print("best action: ",best_action)
        # print("costs of actions: \n",sorted(play_cost,key=lambda x: x[1]))
        return best_action
  