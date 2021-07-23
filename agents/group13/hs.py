from template import Agent
import sys
sys.path.append('agents/group13/')
from .game_utils.mab import *
from .game_utils.game_funcs import *
from .hs_utils.goalstates import *
# from .hs_utils.boardlist import *
from .hs_utils.search_problem import *
import random
import copy

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.N = -1
    
    def SelectAction(self,actions,game_state):
        self.N = self.N + 1
        goalstate = self.InitGoalState(game_state) # get GoalState object
        actions = self.ReduceActions(actions, game_state) # Reducing No. of Possible actions
        player = SearchProblem(player_id=self.id, goal_states=goalstate,
                                actions=actions) # Create search object
        next_action = player.GreedyAlgorithm(heuristic="weighted") # Play local greedy search
        return next_action


    def InitGoalState(self,game_state):
        """
        Create Goalstate object. Return object to be used in SelectAction().
        """
        hand_cards = game_state.agents[self.id].hand
        draft_cards = game_state.board.draft
        r_tiles = game_state.board.plr_coords['r']
        b_tiles = game_state.board.plr_coords['b']
        try:
            if self.id % 2:
                # Initiating GoalState object
                board_state = GoalState(b_tiles,r_tiles,hand_cards,draft_cards) 
            else:
                board_state = GoalState(r_tiles,b_tiles,hand_cards,draft_cards)
        except:
            print("goalstate error")
        return board_state
    
    def ReduceActions(self, actions, game_state):
        """
        Game strategies. Aim is to reduce possible actions inputted into greedy search. 
        """
        # pick double eye jack always
        draft = game_state.board.draft
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
        self_hand = game_state.agents[self.id].hand
        free_coords = game_state.board.empty_coords
        if self.N >= 2:
            self_urgent_place1, self_urgent_place2, oppo_urgent_place1, oppo_urgent_place2 = [], [], [], []
            oppo_urgent_remove1, oppo_urgent_remove2 = [], []
            # (self_color, oppo_color) = ('b', 'r') if self.id % 2 else ('r', 'b')
            # current_seq_self = sum(game_state.agents[i].completed_seqs for i in range(len(game_state.agents)))
            # current_seq_oppo = sum(game_state.agents[i].completed_seqs for i in range(len(game_state.agents)))
            if self.id % 2:
                self_color, oppo_color = 'b', 'r'
                current_seq_self = game_state.agents[1].completed_seqs + game_state.agents[3].completed_seqs
                current_seq_oppo = game_state.agents[0].completed_seqs + game_state.agents[2].completed_seqs
            else:
                self_color, oppo_color = 'r', 'b'
                current_seq_self = game_state.agents[0].completed_seqs + game_state.agents[2].completed_seqs
                current_seq_oppo = game_state.agents[1].completed_seqs + game_state.agents[3].completed_seqs

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

                # remove heart area sometime?
                # else:
                #     j1i = 0
                #     for c in self_hand:
                #         if c == 'jh' or c == 'js':
                #             j1i += 1
                #     if j1i > 1:
                #         for a in actions:
                #             if a['type'] == 'remove' and a['coords'] in [(4, 4), (4, 5), (5, 4), (5, 5)]:
                #                 actions_pro.append(a)
        """
        stratgy end here
        """

        if len(actions_pro) > 0:
            return actions_pro
        elif ('jd' in self_hand) or ('jc' in self_hand) or ('jh' in self_hand) or ('js' in self_hand):
            for a in actions:
                if a['play_card'][0] == 'j':
                    continue
                else:
                    actions_pro.append(a)
        else:
            actions_pro = actions

        # strategy for first 2 round
        if self.N <= 1:
            actions_pro = self.FirstActions(actions_pro)
        elif self.N < 10:
            actions_pro = self.PlayCentroid(actions_pro, game_state.board.plr_coords)
        elif random.randrange(10) < 8:
            actions_pro = self.PlayCentroid(actions_pro, game_state.board.plr_coords)

        return actions_pro

    def FirstActions(self, actions):
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
        centroid_actions = []
        coords = plr_coords['b'] if self.id % 2 else plr_coords['r']
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