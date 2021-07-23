import sys
sys.path.append('agents/group13/')
from template import Agent
import random
import os.path
from .q_utils import qutils
from .q_utils.features import *
from .q_utils.game_function import GameEnds13, CheckSeq13
from .q_utils.goalStates import *
import json
import copy

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.weights = {"bias": 4.154530988995869, "euclideanDistanceCentroid": -0.054836601338983666, "neighbour": -0.06600000390530451, "heart": -0.028027612687064798, "blockHeart": 0.006933994773046544, "eHorizontal": 0.1458311668951386, "eVertical": 0.16439590537220247, "eIandIIIDiag": -0.07251413491052584, "eIIandIVDiag": -0.09679381812983791, "draftHorizontal": 0.29719311237595086, "draftVertical": 0.27793905018540527, "draftDiagIandIII": -0.06433959368738601, "draftDiagIIandIV": -0.022979822531690862, "draftJacks": 0.6652034777563969, "PlayCentre": 0.4696501677800702, "HeuristicValuePlace": 0.7981586178632308, "HeuristicValueDraft": 0.25177106485443485}
        self.gamma = 0.9
        self.action_count = 0
        # id = str(self.id)
        # self.filename = 'Agent0vsdebug2.txt'
        # file_path = './'+self.filename
        # if os.path.exists(file_path):
        #     self.weights = self.read_json(self.filename)
        # else:
        #     print('weights file does not exist')

    def read_json(self, filename):
        with open(filename) as f:
            for line in f:
                pass
            lastWeight = json.loads(line)

        return lastWeight

    def SelectAction(self,actions,game_state):
        """
        Compute the best action to take in a state by getting the Qvalue from multiplying
        weights vector and f(s,a) vectors. Note that if there are no legal actions,
        at the terminal state, return None.
        """

        # reducedActions = self.ReduceActions(actions,game_state)
        # if len(reducedActions) == 0:
        #     reducedActions = actions
        maxQvalue = self.valueFromQvalues(game_state, actions)

        if GameEnds13(game_state):
            return None
        else:
            maxAction = [action for action in actions if self.getQvalue(game_state, action) == maxQvalue]
            best_action = random.choice(maxAction)
        self.action_count += 1
        return best_action

    def valueFromQvalues(self, state, actions):
        """
          Returns max_action Q(state,action) where the max is over legal actions.
        """
        if GameEnds13(state):
            return 0
        else:
            value = max([self.getQvalue(state, action) for action in actions])
        return value

    def getQvalue(self, state, action):
        """
        Calculate the Qvalue using the weight vectors and
        feature vectors return from getFeatures function
        """
        featureVector = self.getFeatures(state, action)
        qValue = 0

        for k in featureVector.keys():
            qValue = qValue + self.weights[k] * featureVector[k]
        return qValue

    def getFeatures(self, state, action):
        """
        Given the state and action build the approximation function features f(s,a)
        """
        features = qutils.Qcounter()
        features['bias'] = 1.0

        if state is None:
            return features
        else:

            if self.id%2 == 0:
                plrCoords = state.board.plr_coords['r']
                oppCoords = state.board.plr_coords['b']
            else:
                plrCoords = state.board.plr_coords['b']
                oppCoords = state.board.plr_coords['r']

            goalState = GoalState(state.board.plr_coords['r'],state.board.plr_coords['b'],state.agents[self.id].hand,
                                  state.board.draft)
            if action['coords'] is not None:
                draftCoords = goalState.CardsToCoords([action['draft_card']])
            else:
                draftCoords = None

            # goalState.PlayAction(action)
            features['euclideanDistanceCentroid'] = eucDist(action, plrCoords)
            features['neighbour'] = neighbour(action, plrCoords, oppCoords)
            features['heart'] = heart(action, plrCoords)
            features['blockHeart'] = blockHeart(action, oppCoords)
            features['eHorizontal'] = eHorizontal(state, action, plrCoords, oppCoords)
            features['eVertical'] = eVertical(state, action, plrCoords, oppCoords)
            features['eIandIIIDiag'] = eIandIIIDiagonal(state, action, plrCoords, oppCoords)
            features['eIIandIVDiag'] = eIIandIVDiagonal(state, action, plrCoords, oppCoords)
            features['draftHorizontal'] = draftHorizontal(state, plrCoords, oppCoords, draftCoords)
            features['draftVertical'] = draftVertical(state, plrCoords, oppCoords, draftCoords)
            features['draftDiagIandIII'] = draftDiagIandIII(state, plrCoords, oppCoords, draftCoords)
            features['draftDiagIIandIV'] = draftDiagIIandIV(state, plrCoords, oppCoords, draftCoords)
            features['draftJacks'] = DraftJacks(action)
            features['PlayCentre'] = PlayCentre(action)
            features['HeuristicValuePlace'] = HeuristicValue(action, goalState)
            features['HeuristicValueDraft'] = HeuristicValueDraft(action, goalState, draftCoords, self.gamma)

        return features

    # def ReduceActions(self, actions, game_state):
    #     draft = game_state.board.draft
    #     self_hand = game_state.agents[self.id].hand
    #     free_coords = game_state.board.empty_coords
    #     if self.id % 2:
    #         self_color, oppo_color = 'b', 'r'
    #         current_seq_self = game_state.agents[1].completed_seqs# + game_state.agents[3].completed_seqs
    #         current_seq_oppo = game_state.agents[0].completed_seqs #+ game_state.agents[2].completed_seqs
    #     else:
    #         self_color, oppo_color = 'r', 'b'
    #         current_seq_self = game_state.agents[0].completed_seqs #+ game_state.agents[2].completed_seqs
    #         current_seq_oppo = game_state.agents[1].completed_seqs #+ game_state.agents[3].completed_seqs
    #
    #     # pick double eye jack always
    #     if ('jd' in draft) or ('jc' in draft):
    #         actions_pro = []
    #         for a in actions:
    #             if a['draft_card'] == 'jd' or a['draft_card'] == 'jc':
    #                 actions_pro.append(a)
    #         actions = actions_pro
    #     elif ('jh' in draft) or ('js' in draft):
    #         actions_pro = []
    #         for a in actions:
    #             if a['draft_card'] == 'jh' or a['draft_card'] == 'js':
    #                 actions_pro.append(a)
    #         actions = actions_pro
    #
    #     """
    #     consider strategy if game has started for a while
    #     """
    #     actions_pro = []
    #     if self.action_count > 1:
    #         self_urgent_place1, self_urgent_place2, oppo_urgent_place1, oppo_urgent_place2 = [], [], [], []
    #         oppo_urgent_remove1, oppo_urgent_remove2 = [], []
    #         # find urgent tile need to place
    #         for coord in free_coords:
    #             # offence
    #             gs_copy1 = copy.deepcopy(game_state)
    #             gs_copy1.board.chips[coord[0]][coord[1]] = self_color
    #             gs_copy1.board.empty_coords.remove(coord)
    #             gs_copy1.board.plr_coords[self_color].append(coord)
    #             seq, _ = CheckSeq13(gs_copy1.board.chips, game_state.agents[self.id], coord)
    #             if seq is not None:
    #                 if seq['num_seq'] > 1:
    #                     self_urgent_place1.append(coord)
    #                 else:
    #                     self_urgent_place2.append(coord)
    #             # defence
    #             gs_copy2 = copy.deepcopy(game_state)
    #             gs_copy2.board.chips[coord[0]][coord[1]] = oppo_color
    #             gs_copy2.board.empty_coords.remove(coord)
    #             gs_copy2.board.plr_coords[oppo_color].append(coord)
    #             seq, _ = CheckSeq13(gs_copy2.board.chips, game_state.agents[(self.id + 1) % len(game_state.agents)], coord)
    #             if seq is not None:
    #                 if seq['num_seq'] > 1 or current_seq_oppo == 1:
    #                     oppo_urgent_place1.append(coord)
    #                     for coord_nbs in seq['coords']:
    #                         coord_nbs.remove(coord)
    #                         oppo_urgent_remove1.extend(coord_nbs)
    #                 else:
    #                     oppo_urgent_place2.append(coord)
    #                     for coord_nbs in seq['coords']:
    #                         coord_nbs.remove(coord)
    #                         oppo_urgent_remove2.extend(coord_nbs)
    #
    #         oppo_urgent_remove1 = list(set(oppo_urgent_remove1))
    #         oppo_urgent_remove2 = list(set(oppo_urgent_remove2))
    #
    #         # check if can place
    #         self_actions_pro1, self_actions_pro2, oppo_actions_pro1, oppo_actions_pro2 = [], [], [], []
    #         if len(self_urgent_place1) > 0 or len(self_urgent_place2) > 0 or len(oppo_urgent_place1) > 0 or len(oppo_urgent_place2) > 0:
    #             for a in actions:
    #                 if a['type'] == 'place':
    #                     if a['coords'] in self_urgent_place1:
    #                         self_actions_pro1.append(a)
    #                     elif a['coords'] in oppo_urgent_place1:
    #                         oppo_actions_pro1.append(a)
    #                     elif a['coords'] in self_urgent_place2:
    #                         self_actions_pro2.append(a)
    #                     elif a['coords'] in oppo_urgent_place2 and a['play_card'][0] != 'j':
    #                         oppo_actions_pro2.append(a)
    #
    #         # check if can place on the urgent
    #         if len(self_actions_pro1) > 0:
    #             actions_pro = self_actions_pro1
    #         elif len(oppo_actions_pro1) > 0:
    #             actions_pro = oppo_actions_pro1
    #         elif len(self_actions_pro2) > 0:
    #             actions_pro = self_actions_pro2
    #         elif len(oppo_actions_pro2) > 0:
    #             actions_pro = oppo_actions_pro2
    #         else:
    #             actions_pro = []
    #
    #         # if 1 eye jack in the hand consider remove
    #         if len(actions_pro) == 0 and (('jh' in self_hand) or ('js' in self_hand)):
    #             urgent_remove0, urgent_remove1, urgent_remove2 = [], [], []
    #             actions_pro0, actions_pro1, actions_pro2 = [], [], []
    #
    #             if len(oppo_urgent_remove1) > 0:
    #                 urgent_remove0 = oppo_urgent_remove1
    #             elif len(oppo_urgent_remove2) > 0:
    #                 urgent_remove0 = oppo_urgent_remove2
    #
    #             for coord in game_state.board.plr_coords[oppo_color]:
    #                 gs_copy = copy.deepcopy(game_state)
    #                 gs_copy.board.chips[coord[0]][coord[1]] = self_color
    #                 gs_copy.board.plr_coords[oppo_color].remove(coord)
    #                 gs_copy.board.plr_coords[self_color].append(coord)
    #                 seq, _ = CheckSeq13(gs_copy.board.chips, game_state.agents[self.id], coord)
    #                 if seq is not None:
    #                     if seq['num_seq'] > 1:
    #                         urgent_remove1.append(coord)
    #                     else:
    #                         urgent_remove2.append(coord)
    #
    #             if len(urgent_remove0) > 0 or len(urgent_remove1) > 0 or len(urgent_remove2) > 0:
    #                 for a in actions:
    #                     if a['type'] == 'remove':
    #                         if a['coords'] in urgent_remove0:
    #                             actions_pro0.append(a)
    #                         elif a['coords'] in urgent_remove1:
    #                             actions_pro1.append(a)
    #                         elif a['coords'] in urgent_remove2:
    #                             actions_pro2.append(a)
    #                 if len(actions_pro0) > 0:
    #                     actions_pro = actions_pro0
    #                     action_rm_center = []
    #                     for a in actions_pro:
    #                         if a['coords'] in [(4, 4), (4, 5), (5, 4), (5, 5)]:
    #                             action_rm_center.append(a)
    #                     if len(action_rm_center) > 0:
    #                         actions_pro = action_rm_center
    #                 elif len(actions_pro1) > 0:
    #                     actions_pro = actions_pro1
    #                 elif len(actions_pro2) > 0:
    #                     actions_pro = actions_pro2
    #                 else:
    #                     actions_pro = []
    #     """
    #     stratgy end here
    #     """
    #
    #     if len(actions_pro) > 0:
    #         return actions_pro
    #     elif ('jd' in self_hand) or ('jc' in self_hand) or ('jh' in self_hand) or ('js' in self_hand):
    #         for a in actions:
    #             if a['type'] != 'trade' and a['play_card'][0] == 'j':
    #                 continue
    #             else:
    #                 actions_pro.append(a)
    #     else:
    #         actions_pro = actions
    #
    #     # strategy for first 2 round
    #     if self.action_count < 2:
    #         actions_pro = self.FirstActions(actions_pro)
    #     elif len(actions_pro) > 15 and random.random() < 0.9:
    #         hs = GoalState(game_state.board.plr_coords[self_color],
    #                        game_state.board.plr_coords[oppo_color],
    #                        self_hand, draft)
    #         actions_prune = hs.PruneActions(actions_pro)
    #         if len(actions_prune):
    #             actions_pro = actions_prune
    #     # elif self.N < 10:
    #     #     actions_pro = self.PlayCentroid(actions_pro, game_state.board.plr_coords)
    #     # elif random.randrange(10) < 8:
    #     #     actions_pro = self.PlayCentroid(actions_pro, game_state.board.plr_coords)
    #
    #     return actions_pro
    #
    #
    # def FirstActions(self, actions):
    #     min_d = 9
    #     first_actions = []
    #     for a in actions:
    #         if a['type'] == 'place':
    #             (r, c) = a['coords']
    #             d = abs(r - 4.5) + abs(c - 4.5)
    #             if d < min_d:
    #                 first_actions = [a]
    #                 min_d = d
    #             elif d == min_d:
    #                 first_actions.append(a)
    #     return first_actions
    #
    #
    # def PlayCentroid(self, actions, plr_coords):
    #     centroid_actions = []
    #     coords = plr_coords['b'] if self.id % 2 else plr_coords['r']
    #     if len(coords) == 0:
    #         x, y = 4.5, 4.5
    #     else:
    #         x, y = 0, 0
    #         for (r, c) in coords:
    #             x += r
    #             y += c
    #         x /= len(coords)
    #         y /= len(coords)
    #     for a in actions:
    #         if a['type'] == 'place' or a['type'] == 'remove':
    #             r, c = a['coords']
    #             if abs(r - x) <= 3 and abs(c - y) <= 3:
    #                 centroid_actions.append(a)
    #         else:
    #             centroid_actions.append(a)
    #     if len(centroid_actions) > 0:
    #         actions = centroid_actions
    #     return actions