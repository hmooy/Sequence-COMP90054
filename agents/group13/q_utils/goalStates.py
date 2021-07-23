import numpy as np
from collections import defaultdict as dd
import math
import pandas as pd
import operator

BOARD = [['jk','2s','3s','4s','5s','6s','7s','8s','9s','jk'],
         ['6c','5c','4c','3c','2c','ah','kh','qh','th','ts'],
         ['7c','as','2d','3d','4d','5d','6d','7d','9h','qs'],
         ['8c','ks','6c','5c','4c','3c','2c','8d','8h','ks'],
         ['9c','qs','7c','6h','5h','4h','ah','9d','7h','as'],
         ['tc','ts','8c','7h','2h','3h','kh','td','6h','2d'],
         ['qc','9s','9c','8h','9h','th','qh','qd','5h','3d'],
         ['kc','8s','tc','qc','kc','ac','ad','kd','4h','4d'],
         ['ac','7s','6s','5s','4s','3s','2s','2h','3h','5d'],
         ['jk','ad','kd','qd','td','9d','8d','7d','6d','jk']]

#Store dict of cards and their coordinates for fast lookup.
COORDS = dd(list)
for row in range(10):
    for col in range(10):
        COORDS[BOARD[row][col]].append((row,col))

class BoardList:
    def __init__(self,plr_tiles,opp_tiles,cards,draft,actions):
        self.plr_tiles = plr_tiles
        self.opp_tiles = opp_tiles
        self.hand_cards = cards
        self.draft_cards = draft
        self.hand_coords = self.CardsToCoords(self.hand_cards)
        self.draft_coords = self.CardsToCoords(self.draft_cards)
        self.two_eyed,self.one_eyed = self.UpdateJacks()
        self.board =  self.ConvertBoard(plr_tiles,opp_tiles)
        self.goalstates = self.GetGoalStates(self.board)
        self.actions= actions

    # def __init__(self,game_state, agent_id, actions):
    #     self.game_state = game_state
    #     if agent_id % 2:
    #         self_color, oppo_color = 'b', 'r'
    #     else:
    #         self_color, oppo_color = 'r', 'b'
    #     self.plr_tiles = game_state.board.plr_coords[self_color]
    #     self.opp_tiles = game_state.board.plr_coords[oppo_color]
    #     self.hand_cards = game_state.agents[agent_id].hand
    #     self.draft_cards = game_state.board.draft
    #     self.hand_coords = self.CardsToCoords(self.hand_cards)
    #     self.draft_coords = self.CardsToCoords(self.draft_cards)
    #     self.two_eyed,self.one_eyed = self.UpdateJacks()
    #     self.board =  self.ConvertBoard(self.plr_tiles,self.opp_tiles)
    #     self.goalstates = self.GetGoalStates(self.board)
    #     self.game_state = game_state
    #     self.actions = actions

    def GetGoalStates(self,board):
        goalstates = [[(4,4),(5,4),(4,5),(5,5)]]
        board = np.array(board)
        for row in board:
            for i in range(len(row)-4):
                goal = row[i:i+5]
                goalstates.append(goal.tolist())

        for row in board:
            for i in range(len(row)-4):
                goal = row[i:i+5]
                goalstates.append(goal.tolist())

        diags = [board[::-1,:].diagonal(i) for i in range(-board.shape[0]+1,board.shape[1])]
        diags.extend(board.diagonal(i) for i in range(board.shape[1]-1,-board.shape[0],-1))
        diag_list = [n.tolist() for n in diags if len(n)>=5]
        for diag in diag_list:
            n = len(diag)-4
            i=0
            while n>0:
                goalstates.append(diag[i:i+5])
                n-=1
                i+=1
        return goalstates

    def InitiateBoard(self):
        x,y=10,10
        board = []
        for i in range(x):
            row = [(i,j) for j in range(y)]
            board.append(row)
        corners = [(0,0),(9,0),(0,9),(9,9)]
        board = self.UpdateBoard(board,corners,1)
        return board

    def ConvertBoard(self, plr_tiles,opp_tiles):
        board = self.InitiateBoard()
        board = self.UpdateBoard(board,plr_tiles,1)
        board = self.UpdateBoard(board,opp_tiles,2)
        return board

    def UpdateBoard(self, board,tiles,val):
        for coord in tiles:
            x,y=coord
            board[x][y] = val
        return board

    def UpdateJacks(self):
        twoEyed = False
        oneEyed = False
        if any(x in self.hand_cards for x in ['jd', 'jc']):
            twoEyed = True
        if any(x in self.hand_cards for x in ['jh', 'js']):
            oneEyed = True
        return twoEyed,oneEyed

    def PruneActions(self):
        total = 0
        costs = []
        for action in self.actions:
            print("log1")
            if action['type'] == "place":
                play_coord = action['coords']
                print("log2")
                d = self.CheckCurrentValue(self.goalstates,play_coord)
                if d:
                    print("log3")
                    h = self.ComputeWeightedHeuristic(d['player_tiles'],d['opponent_tiles'],d['sequence_count'])
                    #h = self.ComputeNearWinHeuristic(d['exclusive_player_tiles'],d['exclusive_goal_count'])
                    total+=h
                else:
                    print("log4")
                    h = math.inf
                costs.append((action,h))
        print("log5")
        costs = sorted(costs,key=lambda x: x[1],reverse=True)
        i = 0
        thresh = 0.1
        while i < len(costs):
            print("log6")
            val = costs[i][1]
            if val/total<thresh:
                print("before removal/n",self.actions)
                self.actions.remove(costs[i][0])
                print("after removal/n",self.actions)
            else:
                break
        return self.actions

    def CheckCurrentValue(self,goalstates,coord):
        d = dd(float)
        for goal in goalstates:
            if coord in goal:
                d['player_tiles']+= math.e**goal.count(1)
                d['opponent_tiles']+= math.e**(goal.count(2)-1)

                # changed name for clarity
                d['sequence_count']+= 1 # Total number of sequences this coordinate exists in
                d['exclusive_goal_count'] += 1 if goal.count(2)==0 else 0 # Total potential goals where there are no opponent chips
                d['exclusive_player_tiles'] += math.e**goal.count(1) if goal.count(2)==0 else 0 # Total player tiles in our sequences without opponent tiles
        return d

    # New heuristic
    def ComputeNearWinHeuristic(self,exclusive_player_tiles,exclusive_goal_count):
        return (exclusive_goal_count/(exclusive_player_tiles+1))

    def ComputeWeightedHeuristic(self, player_tiles,opponent_tiles,sequence_count):
        w1,w2,w3 = 0.4,0.3,0.3
        return (100/(w1*player_tiles+w2*opponent_tiles+w3*sequence_count))

# class BoardList:
#     def __init__(self,plr_tiles,opp_tiles,cards,draft):
#         self.plr_tiles = plr_tiles
#         self.opp_tiles = opp_tiles
#         self.hand_cards = cards
#         self.draft_cards = draft
#         self.hand_coords = self.CardsToCoords(self.hand_cards)
#         self.draft_coords = self.CardsToCoords(self.draft_cards)
#         self.two_eyed,self.one_eyed = self.UpdateJacks()
#         self.board =  self.ConvertBoard(plr_tiles,opp_tiles)
#         self.goalstates = self.GetGoalStates(self.board)

#         # self.hand_cards = cards
#         # self.hand_coords = self.CardsToCoords(self.hand_cards)
#         # self.draft_cards = draft
#         # self.draft_coords = self.CardsToCoords(self.draft_cards)
#         # self.two_eyed,self.one_eyed = self.UpdateJacks()

#     def return_inputs(self):
#         return self.plr_tiles,self.opp_tiles,self.hand_cards,self.draft_cards

#     # def PlayAction(self,action):
#     #     play_coord = action['coords']
#     #     draft_coord = self.CardsToCoords(action['draft_card'])


#     def GetGoalStates(self,board):
#         goalstates = [[(4,4),(5,4),(4,5),(4,5)]]
#         board = np.array(board)
#         for row in board:
#             for i in range(len(row)-4):
#                 goal = row[i:i+5]
#                 goalstates.append(goal.tolist())

#         for row in board:
#             for i in range(len(row)-4):
#                 goal = row[i:i+5]
#                 goalstates.append(goal.tolist())

#         diags = [board[::-1,:].diagonal(i) for i in range(-board.shape[0]+1,board.shape[1])]
#         diags.extend(board.diagonal(i) for i in range(board.shape[1]-1,-board.shape[0],-1))
#         diag_list = [n.tolist() for n in diags if len(n)>=5]
#         for diag in diag_list:
#             n = len(diag)-4
#             i=0
#             while n>0:
#                 goalstates.append(diag[i:i+5])
#                 n-=1
#                 i+=1
#         goalstates = self.CompletedSequences(goalstates)
#         return goalstates

#     def InitiateBoard(self):
#         x,y=10,10
#         board = []
#         for i in range(x):
#             row = [(i,j) for j in range(y)]
#             board.append(row)
#         corners = [(0,0),(9,0),(0,9),(9,9)]
#         board = self.UpdateBoard(board,corners,1)
#         return board

#     def ConvertBoard(self, plr_tiles,opp_tiles):
#         board = self.InitiateBoard()
#         board = self.UpdateBoard(board,plr_tiles,1)
#         board = self.UpdateBoard(board,opp_tiles,2)
#         return board

#     def CompletedSequences(self,goalstates):
#         completed_seq = [3,3,3,3,3]
#         goalstates = [goal if goal.count(1) != 5 else completed_seq for goal in goalstates]
#         return goalstates

#     def UpdateBoard(self, board,tiles,val):
#         for coord in tiles:
#             x,y=coord
#             board[x][y] = val
#         return board

#     def UpdateJacks(self):
#         twoEyed = False
#         oneEyed = False
#         if any(x in self.hand_cards for x in ['jd', 'jc']):
#             twoEyed = True
#         if any(x in self.hand_cards for x in ['jh', 'js']):
#             oneEyed = True
#         return twoEyed,oneEyed



#     ## ===================== COMPUTING HEURISTIC ===================== ##
#     #  FUNCTIONS TO DO:
#     #  * Check if can remove card in our way.
#     #  * Optimise selection when more than 1 of same category
#     #
#     ## =============================================================== ##
#     def GetAction(self):
#         if self.CheckNearWin():
#             print("Returned near win")
#             return self.CheckNearWin()

#         if self.CheckNearLoss():
#             print("Returned near loss")
#             return self.CheckNearLoss()

#         if self.CheckHearts():
#             print("Returned hearts")
#             return self.CheckHearts()

#         if self.one_eyed:
#             if self.CheckInterference():
#                 print("Returned interference")
#                 return self.CheckInterference()

#         d1 = self.CheckCurrentValue(self.goalstates)
#         # d2 = self.CheckPotentialValue()
#         # print("potential")
#         # for k,v in d2.items():
#         #     print(k,": ",dict(v))
#         # print()

#         max_val = 0
#         chosen_card = None
#         for k,v in d1.items():
#             val = self.ComputeHeuristic(v['player_tiles'],v['opponent_tiles'],v['goal_count'])
#             if max_val < val:
#                 max_val = val
#                 chosen_card =k
#         print("Returned chosen card")
#         return 'place',chosen_card

#     def OptimalDraft(self):

#         return 0

#     def CheckHearts(self):
#         for card in [(4,4),(5,4),(4,5),(4,5)]:
#             if card in self.hand_coords:
#                 if self.board[card[0]][card[1]] == card:
#                     return 'place', card
#         return None


#     def CheckNearWin(self):
#         print("Checking near win..")
#         near_wins = [goal for goal in self.goalstates if len(goal)==4 and goal.count(1)==3]+[goal for goal in self.goalstates if goal.count(1) == 4]
#         if near_wins:
#             for card in self.hand_coords:
#                 for goal in near_wins:
#                     if card in goal: #picking first one found
#                         print('winning card: ',card)
#                         print('held cards',self.hand_coords)
#                         print('Goals: ',near_wins)
#                         return 'place', card
#             if self.two_eyed:
#                 print("two-eyed: ",[x for x in near_wins[0] if type(x) == tuple][0])
#                 return 'place',[x for x in near_wins[0] if type(x) == tuple][0]
#         return None

#     def CheckNearLoss(self):
#         print("Checking near loss..")
#         near_loss = [x for x in self.goalstates if len(x)==4 and x.count(2)==3]+[goal for goal in self.goalstates if goal.count(2) == 4]
#         if near_loss:
#             for card in self.hand_coords:
#                 for goal in near_loss:
#                     if card in goal: #picking first one found
#                         return 'place', card
#             if self.one_eyed:
#                 return 'remove',self.FindCoordfromGoal(near_loss[0])  #picking first one found
#             if self.two_eyed:
#                 return 'place',[x for x in near_loss[0] if type(x) == tuple][0]  #picking first one found
#         return None

#     def FindCoordfromGoal(self,goal,val=2):

#         originalstates = self.GetGoalStates(self.InitiateBoard())


#         goal_idx = self.goalstates.index(goal)

#         opponent_tile_idx = goal.index(val) # getting first one, need to optimise this

#         return originalstates[goal_idx][opponent_tile_idx]

#     def CheckInterference(self):
#         print("Checking interference..")
#         # Interference: if we have 3 or more tiles in goal, while opponent is blocking with at least 1
#         # NEED TO UPDATE THIS TO SHOW FOR ALL GOAL STATES.. if still able to make goal, should ignore.
#         # Update to check if we have card for that coordinate

#         near_wins = [goal for goal in self.goalstates if (2<goal.count(1)<5 and 0<goal.count(2))]
#         if near_wins:
#             return 'remove', self.FindCoordfromGoal(near_wins[0])
#         return None

#     def CheckCurrentValue(self,goalstates):
#         d = dd(lambda: dd(float))
#         for goal in goalstates:
#             for card in self.hand_coords:
#                 if card in goal:
#                     d[card]['player_tiles']+= math.e**goal.count(1)
#                     d[card]['opponent_tiles']+= math.e**(goal.count(2)-1)
#                     d[card]['goal_count']+= 1
#         return d

#     def ComputeHeuristic(self, player_tiles,opponent_tiles,goal_count):
#         w1,w2,w3 = .4,.3,.3
#         return(w1*player_tiles+w2*opponent_tiles+w3*goal_count)

#     # def CheckPotentialValue(self):
#     #     d = dd(lambda: dd(int))
#     #     for card in self.hand_coords:
#     #         print("card: ",card)
#     #         hand_cards = self.hand_coords.copy()
#     #         hand_cards.remove(card)
#     #         print(self.board.copy())
#     #         potentialGoalState = self.GetGoalStates(self.UpdateBoard(self.board.copy(),hand_cards,1))
#     #         for goal in potentialGoalState:
#     #             if card in goal:
#     #                 d[card]['player_tiles']+= goal.count(1)
#     #                 d[card]['opponent_tiles']+= goal.count(2)
#     #                 d[card]['goal_count']+= 1
#     #     return d




#     # def ComputeHeuristc(self,cards):
#     #     # Check if can win immediately
#     #     # try with normal card, then 2 eyed jack if have
#     #     coords = self.CardsToCoords(cards)
#     #     result = self.NearEnd(coords)
#     #     if result:
#     #         return result
#     #     # Check if can block opponent
#     #     result = self.NearEnd(coords,False)
#     #     if result:
#     #         return result


#     #     d = dd(lambda: dd(int))
#     #     for goal in self.goalstates:

#     #         for card in coords:
#     #             if card in goal:
#     #                 d[card]['player_tiles']+= goal.count(1)
#     #                 d[card]['opponent_tiles']+= goal.count(2)
#     #                 d[card]['goal_count']+= 1
#     #     highest = 0
#     #     card = None
#     #     for k,data in d.items():
#     #         average = data['player_tiles']+data['opponent_tiles']+data['goal_count']
#     #         if average > highest:
#     #             highest = average
#     #             card = k
#     #     return 'place',card

#     def CardsToCoords(self,cards):
#         hand_coords = []
#         for card in cards:
#             hand_coords= hand_coords+COORDS[card]
#         return hand_coords

#     def CoordsToCards(self,coords):
#         x,y = coords
#         return BOARD[x][y]

## ======================= NEW CLASS, SINGLE CARD ======================= ##
#
#
## ====================================================================== ##


class GoalState:
    def __init__(self,plr_tiles,opp_tiles,cards,draft):
        self.plr_tiles = plr_tiles
        self.opp_tiles = opp_tiles
        self.hand_cards = cards
        self.draft_cards = draft
        self.hand_coords = self.CardsToCoords(self.hand_cards)
        self.draft_coords = self.CardsToCoords(self.draft_cards)
        self.two_eyed,self.one_eyed = self.UpdateJacks()
        self.board =  self.ConvertBoard(plr_tiles,opp_tiles)
        self.goalstates = self.GetGoalStates(self.board)
    # def CompileGoalStates(self, board):
    #     goalstates = self.GetGoalStates(board)
    #     completed_seq = self.CompletedSequences(goalstates)
    #     if completed_seq:
    #         for coord in completed_seq:
    #             board[coord[x]]

    def GetGoalStates(self,board):
        goalstates = [[(4,4),(5,4),(4,5),(5,5)]]
        board = np.array(board)
        for row in board:
            for i in range(len(row)-4):
                goal = row[i:i+5]
                goalstates.append(goal.tolist())

        for row in board:
            for i in range(len(row)-4):
                goal = row[i:i+5]
                goalstates.append(goal.tolist())

        diags = [board[::-1,:].diagonal(i) for i in range(-board.shape[0]+1,board.shape[1])]
        diags.extend(board.diagonal(i) for i in range(board.shape[1]-1,-board.shape[0],-1))
        diag_list = [n.tolist() for n in diags if len(n)>=5]
        for diag in diag_list:
            n = len(diag)-4
            i=0
            while n>0:
                goalstates.append(diag[i:i+5])
                n-=1
                i+=1
        # self.CompletedSequences(goalstates)
        return goalstates

    # def CompletedSequences(self,goalstates):
    #     try:
    #         completed_seq_idx = goalstates.index([1,1,1,1,1])
    #     except ValueError:
    #         completed_seq_idx = None
    #     if completed_seq_idx:
    #         print("found sequence")
    #         originalstates = self.GetGoalStates(self.InitiateBoard())
    #         print(originalstates)
    #         completed_seq_coords = originalstates[completed_seq_idx]
    #         for coord in completed_seq_coords:
    #             gamestate
    #             print(coord)


    #     return completed_seq_coords

    # return None

    def InitiateBoard(self):
        x,y=10,10
        board = []
        for i in range(x):
            row = [(i,j) for j in range(y)]
            board.append(row)
        corners = [(0,0),(9,0),(0,9),(9,9)]
        board = self.UpdateBoard(board,corners,1)
        return board

    def ConvertBoard(self, plr_tiles,opp_tiles):
        board = self.InitiateBoard()
        board = self.UpdateBoard(board,plr_tiles,1)
        board = self.UpdateBoard(board,opp_tiles,2)
        return board

    def UpdateBoard(self, board,tiles,val):
        for coord in tiles:
            x,y=coord
            board[x][y] = val
        return board

    def UpdateJacks(self):
        twoEyed = False
        oneEyed = False
        if any(x in self.hand_cards for x in ['jd', 'jc']):
            twoEyed = True
        if any(x in self.hand_cards for x in ['jh', 'js']):
            oneEyed = True
        return twoEyed,oneEyed


    ## ===================== COMPUTING HEURISTIC ===================== ##
    #  FUNCTIONS TO DO:
    #  * Check if can remove card in our way.
    #  * Optimise selection when more than 1 of same category
    #
    ## =============================================================== ##
    def PruneActions(self,actions):
        total = 0
        costs = []
        actions_copy = actions.copy()
        for action in actions_copy:
            if action['type'] == "place":
                play_coord = action['coords']
                d = self.CheckCurrentValue(self.goalstates,play_coord)
                if d:
                    h = self.ComputeWeightedHeuristic(d['player_tiles'],d['opponent_tiles'],d['sequence_count'])
                    #h = self.ComputeNearWinHeuristic(d['exclusive_player_tiles'],d['exclusive_goal_count'])
                    total+=h
                else:
                    h = math.inf
                costs.append((action,h))
        costs = sorted(costs,key=lambda x: x[1],reverse=True)
        thresh = 0.1
        for row in costs:
            val = row[1]
            action = row[0]
            # print(action)
            # print("value: ",val)
            # print("total: ",total)
            # print("threshold: ",thresh)
            if (val/total)<thresh:
                if action in actions_copy:
                    actions_copy.remove(action)
            else:
                break
        return actions_copy

    def QlearningAction(self, action_coord):
        if self.CheckNearWin(action_coord):
            return 1
        if self.CheckNearLoss(action_coord):
            return 1
        if self.CheckHearts(action_coord):
            return 1

        d = self.CheckCurrentValue(self.goalstates,action_coord)
        if d:
            h = self.ComputeNearWinHeuristic(d['exclusive_player_tiles'],d['exclusive_goal_count'])
            if h > 1:
                h = 1
        else:
            h = 0
        if h > 1:
            print("heuristic greater than 1: ",h)
        return h


    def PlayAction(self,action):
        play_coord = action['coords']
        draft_coord = self.CardsToCoords(action['draft_card'])
        action_type = action['type']
        if action_type == "place":
            ## These should be checked before entering into search ##
            if self.CheckNearWin(play_coord):
                return 0
            if self.CheckNearLoss(play_coord):
                return 0
            if self.CheckHearts(play_coord):
                return 0

            d = self.CheckCurrentValue(self.goalstates,play_coord)
            # if d:
            #     print('checking weighted heuristic')
            #     h = self.ComputeWeightedHeuristic(d['player_tiles'],d['opponent_tiles'],d['sequence_count'])
            # else:
            #     h=math.inf
            # # return h

            ## NEW HEURISTIC EXAMPLE:
            if d:
                try:
                    h = self.ComputeNearWinHeuristic(d['exclusive_player_tiles'],d['exclusive_goal_count'])
                except:
                    print(d['exclusive_player_tiles']," and ",d['exclusive_goal_count'])
            else:
                h = math.inf
            return h
        elif action_type == "remove":
            # TO DO
            self.CheckInterference(play_coord)
        return math.inf


    def CheckHearts(self,coord):
        if coord in [(4,4),(5,4),(4,5),(5,5)]:
            # print("checking hearts..")
            if self.board[coord[0]][coord[1]] == coord:
                return True
        return None

    def CheckNearWin(self,coord):
        near_wins = [goal for goal in self.goalstates if len(goal)==4 and goal.count(1)==3]+[goal for goal in self.goalstates if goal.count(1) == 4]
        if near_wins:
            # print("Checking near win..")
            for goal in near_wins:
                if coord in goal: #picking first one found
                    return True
        return None

    def CheckInterference(self,coord):
        # print("Checking interference..")
        # Interference: if we have 3 or more tiles in goal, while opponent is blocking with at least 1
        # NEED TO UPDATE THIS TO SHOW FOR ALL GOAL STATES.. if still able to make goal, should ignore.
        # Update to check if we have card for that coordinate

        near_wins = [goal for goal in self.goalstates if (2<goal.count(1)<5 and 0<goal.count(2)<3)]
        if near_wins:
            return 'remove', self.FindCoordfromGoal(near_wins[0])
        return None

    def CheckNearLoss(self,coord):
        near_loss = [x for x in self.goalstates if len(x)==4 and x.count(2)==3]+[goal for goal in self.goalstates if goal.count(2) == 4]
        if near_loss:
            # print("Checking near loss..")
            for goal in near_loss:
                if coord in goal: #picking first one found
                    return True
        return None

    def FindCoordfromGoal(self,goal,val=2):
        originalstates = self.GetGoalStates(self.InitiateBoard())
        goal_idx = self.goalstates.index(goal)
        opponent_tile_idx = goal.index(val) # getting first one, need to optimise this
        return originalstates[goal_idx][opponent_tile_idx]

    def CheckCurrentValue(self,goalstates,coord):
        d = dd(float)
        for goal in goalstates:
            if coord in goal:
                d['player_tiles']+= math.e**goal.count(1)
                d['opponent_tiles']+= math.e**(goal.count(2)-1)

                # changed name for clarity
                d['sequence_count']+= 1 # Total number of sequences this coordinate exists in
                d['exclusive_goal_count'] += 1 if goal.count(2)==0 else 0 # Total potential goals where there are no opponent chips
                d['exclusive_player_tiles'] += math.e**goal.count(1) if goal.count(2)==0 else 0 # Total player tiles in our sequences without opponent tiles
        return d

    # New heuristic
    def ComputeNearWinHeuristic(self,exclusive_player_tiles,exclusive_goal_count):
        return (0.5*exclusive_player_tiles/(exclusive_goal_count+1))

    def ComputeWeightedHeuristic(self, player_tiles,opponent_tiles,sequence_count):
        w1,w2,w3 = 0.4,0.3,0.3
        return (100/(w1*player_tiles+w2*opponent_tiles+w3*sequence_count))

    def CardsToCoords(self,cards):
        hand_coords = []
        for card in cards:
            hand_coords= hand_coords+COORDS[card]
        return hand_coords

    def CoordsToCards(self,coords):
        x,y = coords
        return BOARD[x][y]