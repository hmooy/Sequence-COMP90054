import copy
from .game_funcs import GenerateSuccessor13,GetLegalActions13
from ..hs_utils.goalstates import GoalState
from template import GameState, GameRule
from Sequence.sequence_utils import *
from collections import defaultdict
import random


class State:
    def __init__(self, player_id, game_state, pre_move):
        self.pre_move = pre_move
        self.player_id = player_id
        self.game_state = game_state
        # N
        self.N = 0
        # V value
        self.win_scores_sum = [0, 0, 0, 0]


class Node:
    def __init__(self, state, parent):
        self.state = state
        self.children = []
        self.parent = parent

    def ExpandChildren(self, actions=None):
        """
        Expand all the valid children of the node, use pruning to reduce the possible actions chosen.
        """
        next_player = (self.state.player_id + 1) % len(self.state.game_state.agents)
        if actions is None:
            actions = GetLegalActions13(self.state.game_state, self.state.player_id)
            actions = _simplyMoves(self.state.game_state, actions, self.state.player_id)
        for action in actions:
            next_gs = GenerateSuccessor13(copy.deepcopy(self.state.game_state), action, self.state.player_id)
            self.children.append(Node(State(next_player, next_gs, action), self))

def _simplyMoves(game_state, actions, player_id):
    """Cut down action list, remove all actions using functional cards"""
    ans = []
    for a in actions:
        if a['play_card'] is not None and a['play_card'][0] == 'j':
            continue
        ans.append(a)

    # if len(ans) > 15:
    #     if player_id % 2:
    #         self_color, oppo_color = 'b', 'r'
    #     else:
    #         self_color, oppo_color = 'r', 'b'
    #     draft = game_state.board.draft
    #     self_hand = game_state.agents[player_id].hand
    #     hs = GoalState(game_state.board.plr_coords[self_color],
    #                    game_state.board.plr_coords[oppo_color],
    #                    self_hand, draft)
    #
    #     prune_draft = hs.PruneDraftActions(ans)
    #     prune_play = hs.PrunePlayActions(ans)
    #
    #     actions_prune_draft = [action for action in ans if action['draft_card'] in prune_draft]
    #     prune_play_draft = [draft for draft in actions_prune_draft for play in prune_play if
    #                         draft['play_card'] == play['play_card']]
    #
    #     if len(prune_play_draft):
    #         ans = prune_play_draft

    random.shuffle(ans)
    if len(ans) > 60:
        ans = ans[:round(len(ans)/3)]
    elif len(ans) >= 40:
        ans = ans[:round(len(ans)/2)]

    return ans
