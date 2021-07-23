from template import Agent
import sys
sys.path.append('agents/group13/')
from .game_utils import mcts
from .game_utils.mab import *
from .game_utils.q_func import AverageQfunc, AggressiveQfunc
from .game_utils.game_funcs import SimulateGameState, GenerateSuccessor13, CheckSeq13
from .hs_utils.goalstates import GoalState
import random
import time
import copy


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.mab = UCB(0.5)
        self.hands_info = [[], [], []]
        self.discards_info = []
        self.N = -1

    def SelectAction(self, actions, game_state):
        """
        @params:
        actions = list of all possible actions in dictionary format
        game_state = object of SequenceState containing current game state info

        Reduce as many actions to consider as possible, then create the MCTS object
        to run through simulations. 
        """
        begin_time = time.time()
        self.N = self.N + 1
        game_state, self.hands_info = SimulateGameState(gs_copy=copy.deepcopy(game_state),
                                                        agent_id=self.id,
                                                        hands_info=self.hands_info,
                                                        discards_info=self.discards_info)
        self.discards_info = game_state.deck.discards
        actions = self.ReduceActions(actions, game_state)
        if len(actions) == 1:
            return actions[0]
        mcts_player = mcts.MCTS(player_id=self.id, game_state=game_state, actions=actions, time_limit=0.95, mab=self.mab, gamma=0.95)
        cost_time = time.time() - begin_time

        return mcts_player.FindNextMove(AverageQfunc, AggressiveQfunc, time_limit=0.95 - cost_time)

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
            current_seq_self = game_state.agents[1].completed_seqs + game_state.agents[3].completed_seqs
            current_seq_oppo = game_state.agents[0].completed_seqs + game_state.agents[2].completed_seqs
        else:
            self_color, oppo_color = 'r', 'b'
            current_seq_self = game_state.agents[0].completed_seqs + game_state.agents[2].completed_seqs
            current_seq_oppo = game_state.agents[1].completed_seqs + game_state.agents[3].completed_seqs

        ## pick double eye/single eye jack always ##
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

        
        ## Execute the following strategies if not in the first 2 rounds. ##
        actions_pro = []
        if self.N > 1:
            self_urgent_place1, self_urgent_place2, oppo_urgent_place1, oppo_urgent_place2 = [], [], [], []
            oppo_urgent_remove1, oppo_urgent_remove2 = [], []
            # find urgent tile need to place
            for coord in free_coords:
                # offence
                # find any tile for self to finish a sequence
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
                # find any tile for opponent to finish a sequence
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

            # the neighbours of opponents' urgent tile
            oppo_urgent_remove1 = list(set(oppo_urgent_remove1))
            oppo_urgent_remove2 = list(set(oppo_urgent_remove2))

            # check if can place the urgent tiles
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

            # If holding 1-eyed Jack, remove opponent tile
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
        stratgy ends here
        """

        # if no meaning to use the double eye jack and single eye jack, remove them from the hands
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

        # strategy for first 2 round, place in the middle
        if self.N < 2:
            actions_pro = self.FirstActions(actions_pro)
        # strategy for later rounds, prune the acitons using heuristic
        elif len(actions_pro) > 15 and random.random() < 0.9:
            hs = GoalState(game_state.board.plr_coords[self_color],
                           game_state.board.plr_coords[oppo_color],
                           self_hand, draft)
            prune_draft = hs.PruneDraftActions(actions_pro)
            actions_prune_draft = [action for action in actions_pro if action['draft_card'] in prune_draft]

            if len(actions_prune_draft):
                actions_pro = actions_prune_draft

        ## prune using centroid
        # elif self.N < 10:
        #     actions_pro = self.PlayCentroid(actions_pro, game_state.board.plr_coords)
        # elif random.randrange(10) < 8:
        #     actions_pro = self.PlayCentroid(actions_pro, game_state.board.plr_coords)

        return actions_pro

    def FirstActions(self, actions):
        """
        @params:
        actions = list of all possible actions in dictionary format

        Choose the actions that place the chips near the middle of the board.
        """
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
        @params:
        actions = list of all possible actions in dictionary format
        plr_coords = the coodinates of different teams on the board


        Calculate the centroid of self existing chips on the board,
        remove the action that put chips far away from the centroid.
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