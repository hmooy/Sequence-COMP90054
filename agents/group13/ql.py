from template import Agent
import sys
sys.path.append('agents/group13/')
from .q_utils import qutils
from .q_utils.game_function import GameEnds13, GetReward, GenerateSuccessor13, CheckSeq13
import random
import math
import json
import os.path
from .q_utils.features import *
from .q_utils.goalStates import *
import copy


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.s = None
        self.alpha = 0.01
        self.gamma = 0.9
        self.epsilon = 0.2
        self.action = None
        self.action_count = 0
        self.r0 = 0
        self.r1 = 0
        self.r = 0
        self.t = False
        # self.r2 = 0
        # self.r3 = 0

        id = str(self.id)
        self.filename = 'Agent'+id+'MCTS.txt'
        file_path = './'+self.filename
        if os.path.exists(file_path):
            self.weights = self.read_json(self.filename)
        else:
            print('weights file does not exist')
            self.weights = qutils.Qcounter()
        # print(self.weights)

    def SelectAction(self, actions, game_state):
        """
        @params:
        actions = list of all possible actions in dictionary format
        game_state = object of SequenceState containing current game state info

        Reduced actions based on strategy, update weights from previous
        to current state if no agents form a sequence. Pick action using
        MAB algorithm. Change previous state to current state and return
        the action.
        """
        reducedActions = self.ReduceActions(actions,game_state)
        if len(reducedActions) == 0:
            reducedActions = actions

        if self.s is not None or self.t is False:
            self.update(game_state, reducedActions, self.r)
            if self.r == 1 or -1:
                self.t = True

        action = self.EpsGreedy(reducedActions, game_state)
        self.r = self.rewards(game_state, action)

        self.s = game_state
        self.action = action
        self.action_count += 1

        if self.id == 0:
            self.write_json(self.filename)
        elif self.id == 2:
            self.write_json(self.filename)

        return self.action

    def write_json(self, filename):
        """
        Store weights out to txt file
        """
        with open(filename, 'a+') as f:
            f.write(json.dumps(self.weights))
            f.write("\n")

    def read_json(self, filename):
        """
        Read last weights on txt file
        """
        with open(filename) as f:
            for line in f:
                pass
            lastWeight = json.loads(line)

        return lastWeight

    def EpsGreedy(self, actions, game_state):
        """
        Epsilon greedy, explore actions with probability epsilon
        """
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            return self.best_action(actions, game_state)

    def update(self, s_p, actions, reward):
        """
        Update weights based on w' = w + alpha(reward + gamma*max(Q(s',a')) - Q(s,a))f(s,a)
        """
        qValueCurrent = self.getQvalue(self.s, self.action)
        feature = self.getFeatures(self.s, self.action)
        qValue_p = self.valueFromQvalues(s_p, actions)
        # reward = self.rewards(s_p)
        diff = (reward + self.gamma*qValue_p)-qValueCurrent
        for k in feature.keys():
            self.weights[k] = self.weights[k] + self.alpha*diff*feature[k]

    def rewards(self, s_p, action):
        """
        Return completed sequence as reward +1 if self completed sequence
        Otherwise -1 if other agent completed a sequence
        """
        r0 = GenerateSuccessor13(s_p, action, self.id).agents[self.id].completed_seqs
        r1 = GetReward(s_p)
        if self.r0 < r0:
            self.r0 = r0
            # print("0",self.r0)
            return self.r0
        elif self.r1 > (r1*(-1)):
            self.r1 = r1*(-1)
            # print("1",self.r1)
            return self.r1
        else:
            return 0

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

    def valueFromQvalues(self, state, actions):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.
        """

        if GameEnds13(state):
            return 0
        else:
            value = max([self.getQvalue(state, action) for action in actions])
        return value

    def best_action(self, actions, state):
        """
          Compute the best action to take in a state. Note that if there
          are no legal actions, at the terminal state, return None.
        """

        maxQvalue = self.valueFromQvalues(state, actions)

        if GameEnds13(state):
            return None
        else:
            maxAction = [action for action in actions if self.getQvalue(state, action) == maxQvalue]
            best_action = random.choice(maxAction)
        return best_action

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


    def ReduceActions(self, actions, game_state):
        """
        @params:
        actions = list of all possible actions in dictionary format
        game_state = object of SequenceState containing current game state info

        Includes a series of functions to execute game strategies specific to Sequences.  
        Mostly involves reducing the number of possible actions to play. 
        """
        draft = game_state.board.draft
        self_hand = game_state.agents[self.id].hand
        free_coords = game_state.board.empty_coords
        if self.id % 2:
            self_color, oppo_color = 'b', 'r'
            current_seq_self = game_state.agents[1].completed_seqs# + game_state.agents[3].completed_seqs
            current_seq_oppo = game_state.agents[0].completed_seqs #+ game_state.agents[2].completed_seqs
        else:
            self_color, oppo_color = 'r', 'b'
            current_seq_self = game_state.agents[0].completed_seqs #+ game_state.agents[2].completed_seqs
            current_seq_oppo = game_state.agents[1].completed_seqs #+ game_state.agents[3].completed_seqs

        # pick double eye jack always
        if ('jd' in draft) or ('jc' in draft):
            actions_pro = []
            for a in actions:
                if a['draft_card'] == 'jd' or a['draft_card'] == 'jc':
                    actions_pro.append(a)
            actions = actions_pro
        elif ('jh' in draft) or ('js' in draft):
            actions_pro = []
            for a in actions:
                if a['draft_card'] == 'jh' or a['draft_card'] == 'js':
                    actions_pro.append(a)
            actions = actions_pro

        """
        consider strategy if game has started for a while
        """
        actions_pro = []
        if self.action_count > 1:
            self_urgent_place1, self_urgent_place2, oppo_urgent_place1, oppo_urgent_place2 = [], [], [], []
            oppo_urgent_remove1, oppo_urgent_remove2 = [], []
            # find urgent tile need to place
            for coord in free_coords:
                # offence
                gs_copy1 = copy.deepcopy(game_state)
                gs_copy1.board.chips[coord[0]][coord[1]] = self_color
                gs_copy1.board.empty_coords.remove(coord)
                gs_copy1.board.plr_coords[self_color].append(coord)
                seq, _ = CheckSeq13(gs_copy1.board.chips, game_state.agents[self.id], coord)
                if seq is not None:
                    if seq['num_seq'] > 1:
                        self_urgent_place1.append(coord)
                    else:
                        self_urgent_place2.append(coord)
                # defence
                gs_copy2 = copy.deepcopy(game_state)
                gs_copy2.board.chips[coord[0]][coord[1]] = oppo_color
                gs_copy2.board.empty_coords.remove(coord)
                gs_copy2.board.plr_coords[oppo_color].append(coord)
                seq, _ = CheckSeq13(gs_copy2.board.chips, game_state.agents[(self.id + 1) % len(game_state.agents)], coord)
                if seq is not None:
                    if seq['num_seq'] > 1 or current_seq_oppo == 1:
                        oppo_urgent_place1.append(coord)
                        for coord_nbs in seq['coords']:
                            coord_nbs.remove(coord)
                            oppo_urgent_remove1.extend(coord_nbs)
                    else:
                        oppo_urgent_place2.append(coord)
                        for coord_nbs in seq['coords']:
                            coord_nbs.remove(coord)
                            oppo_urgent_remove2.extend(coord_nbs)

            oppo_urgent_remove1 = list(set(oppo_urgent_remove1))
            oppo_urgent_remove2 = list(set(oppo_urgent_remove2))

            # check if can place
            self_actions_pro1, self_actions_pro2, oppo_actions_pro1, oppo_actions_pro2 = [], [], [], []
            if len(self_urgent_place1) > 0 or len(self_urgent_place2) > 0 or len(oppo_urgent_place1) > 0 or len(oppo_urgent_place2) > 0:
                for a in actions:
                    if a['type'] == 'place':
                        if a['coords'] in self_urgent_place1:
                            self_actions_pro1.append(a)
                        elif a['coords'] in oppo_urgent_place1:
                            oppo_actions_pro1.append(a)
                        elif a['coords'] in self_urgent_place2:
                            self_actions_pro2.append(a)
                        elif a['coords'] in oppo_urgent_place2 and a['play_card'][0] != 'j':
                            oppo_actions_pro2.append(a)

            # check if can place on the urgent
            if len(self_actions_pro1) > 0:
                actions_pro = self_actions_pro1
            elif len(oppo_actions_pro1) > 0:
                actions_pro = oppo_actions_pro1
            elif len(self_actions_pro2) > 0:
                actions_pro = self_actions_pro2
            elif len(oppo_actions_pro2) > 0:
                actions_pro = oppo_actions_pro2
            else:
                actions_pro = []

            # if 1 eye jack in the hand consider remove
            if len(actions_pro) == 0 and (('jh' in self_hand) or ('js' in self_hand)):
                urgent_remove0, urgent_remove1, urgent_remove2 = [], [], []
                actions_pro0, actions_pro1, actions_pro2 = [], [], []

                if len(oppo_urgent_remove1) > 0:
                    urgent_remove0 = oppo_urgent_remove1
                elif len(oppo_urgent_remove2) > 0:
                    urgent_remove0 = oppo_urgent_remove2

                for coord in game_state.board.plr_coords[oppo_color]:
                    gs_copy = copy.deepcopy(game_state)
                    gs_copy.board.chips[coord[0]][coord[1]] = self_color
                    gs_copy.board.plr_coords[oppo_color].remove(coord)
                    gs_copy.board.plr_coords[self_color].append(coord)
                    seq, _ = CheckSeq13(gs_copy.board.chips, game_state.agents[self.id], coord)
                    if seq is not None:
                        if seq['num_seq'] > 1:
                            urgent_remove1.append(coord)
                        else:
                            urgent_remove2.append(coord)

                if len(urgent_remove0) > 0 or len(urgent_remove1) > 0 or len(urgent_remove2) > 0:
                    for a in actions:
                        if a['type'] == 'remove':
                            if a['coords'] in urgent_remove0:
                                actions_pro0.append(a)
                            elif a['coords'] in urgent_remove1:
                                actions_pro1.append(a)
                            elif a['coords'] in urgent_remove2:
                                actions_pro2.append(a)
                    if len(actions_pro0) > 0:
                        actions_pro = actions_pro0
                        action_rm_center = []
                        for a in actions_pro:
                            if a['coords'] in [(4, 4), (4, 5), (5, 4), (5, 5)]:
                                action_rm_center.append(a)
                        if len(action_rm_center) > 0:
                            actions_pro = action_rm_center
                    elif len(actions_pro1) > 0:
                        actions_pro = actions_pro1
                    elif len(actions_pro2) > 0:
                        actions_pro = actions_pro2
                    else:
                        actions_pro = []
        """
        stratgy end here
        """

        if len(actions_pro) > 0:
            return actions_pro
        elif ('jd' in self_hand) or ('jc' in self_hand) or ('jh' in self_hand) or ('js' in self_hand):
            for a in actions:
                if a['type'] != 'trade' and a['play_card'][0] == 'j':
                    continue
                else:
                    actions_pro.append(a)
        else:
            actions_pro = actions

        # strategy for first 2 round
        if self.action_count < 2:
            actions_pro = self.FirstActions(actions_pro)
        elif len(actions_pro) > 15 and random.random() < 0.9:
            hs = GoalState(game_state.board.plr_coords[self_color],
                           game_state.board.plr_coords[oppo_color],
                           self_hand, draft)
            actions_prune = hs.PruneActions(actions_pro)
            if len(actions_prune):
                actions_pro = actions_prune
        # elif self.N < 10:
        #     actions_pro = self.PlayCentroid(actions_pro, game_state.board.plr_coords)
        # elif random.randrange(10) < 8:
        #     actions_pro = self.PlayCentroid(actions_pro, game_state.board.plr_coords)

        return actions_pro

    def FirstActions(self, actions):
        '''
        Chip closest to the middle of the board will be played first
        '''
        min_d = 9
        first_actions = []
        for a in actions:
            if a['type'] == 'place':
                (r, c) = a['coords']
                d = abs(r - 4.5) + abs(c - 4.5)
                if d < min_d:
                    first_actions = [a]
                    min_d = d
                elif d == min_d:
                    first_actions.append(a)
        return first_actions

    def PlayCentroid(self, actions, plr_coords):
        """
        Aim to place chips close to the centroid of chips on board
        """
        centroid_actions = []
        coords = plr_coords['b'] if self.id % 2 else plr_coords['r']
        if len(coords) == 0:
            x, y = 4.5, 4.5
        else:
            x, y = 0, 0
            for (r, c) in coords:
                x += r
                y += c
            x /= len(coords)
            y /= len(coords)
        for a in actions:
            if a['type'] == 'place' or a['type'] == 'remove':
                r, c = a['coords']
                if abs(r - x) <= 3 and abs(c - y) <= 3:
                    centroid_actions.append(a)
            else:
                centroid_actions.append(a)
        if len(centroid_actions) > 0:
            actions = centroid_actions
        return actions



