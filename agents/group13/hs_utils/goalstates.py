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


class GoalState:
    """
    @params:
    plr_titles = list of titles on board belonging to player (in coordinate format)
    opp_titles = list of titles on board belonging to opponent (in coordinate format)
    cards = list of cards held by player at current game state (in str format)
    draft = list of cards available to draft at current game state (in str format)

    Class to create list of all goalstates according to current state on board.
    "1" represents player's own tile, "2" represents opponent tile.
    This is used to calculate heuristics for actions.
    """
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

    def GetGoalStates(self,board):
        """ 
        @params:
        board = matrix board representation of current state, as provided by ConvertBoard() function.

        Takes in altered board state and converts it to
        a list of lists of all the possible goalstates. 
        """
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
        """
        Creates a board of coordinates for easy matching.
        """
        x,y=10,10
        board = []
        for i in range(x):
            row = [(i,j) for j in range(y)]
            board.append(row)
        corners = [(0,0),(9,0),(0,9),(9,9)]
        board = self.UpdateBoard(board,corners,1)
        return board

    def ConvertBoard(self, plr_tiles,opp_tiles):
        """
        Initiates board with coordinates, and updates them according 
        to the titles currently placed on the board. 
        """

        board = self.InitiateBoard()
        board = self.UpdateBoard(board,plr_tiles,1)
        board = self.UpdateBoard(board,opp_tiles,2)
        return board
        
    def UpdateBoard(self, board,tiles,val):
        """
        updates board with value if tile exists on the coordinate.
        """
        for coord in tiles:
            x,y=coord
            board[x][y] = val
        return board

    def UpdateJacks(self):
        """
        Updates parameters to inform if the hand contains special action cards.
        """
        twoEyed = False
        oneEyed = False
        if any(x in self.hand_cards for x in ['jd', 'jc']):
            twoEyed = True
        if any(x in self.hand_cards for x in ['jh', 'js']):
            oneEyed = True
        return twoEyed,oneEyed

    ## ===================== COMPUTING HEURISTIC ===================== ##
    #  Functions for computing heuristic. Using in QL, MCTS, and Greedy search
    #  * Different for each Search algorithm
    #  * Used for both play actions and draft cards.
    ## =============================================================== ##
    def PruneDraftActions(self,actions):
        """
        @oarams:
        actions = list of actions in dictionary format

        Takes set of actions, used for MCTS.
        Will return a list of draft cards to simulate.
        """
        total = 0
        costs = []
        actions_copy = actions.copy()
        draft_set = []

        for action in actions_copy:
            if action["draft_card"] in draft_set or action["draft_card"] == None:
                continue
            draft_set.append(action["draft_card"])
            h = self.DraftAction(action)
            costs.append((action['draft_card'],h))
            total+=h
        costs = sorted(costs,key=lambda x: x[1])
        thresh = 0.2
        total_n = len(draft_set)
        final_actions = []
        if total == 0:
            return draft_set
        for card in costs:
            val = card[1]
            if val/total < thresh:
                final_actions.append(card[0])
            else:
                break;
        pruned_n = total_n-len(final_actions)
        print("PruneDraftActions pruned {} actions".format(str(pruned_n)))
        if len(draft_set)==0:
            print("no drafting left. increase thresh")
            return [costs[-1][0]]
        return final_actions
    
    def PrunePlayActions(self,actions):
        """
        @params:
        actions = list of actions in dictionary format

        Takes set of actions, used for MCTS.
        Will return a list of actions to simulate. 
        """
        total = 0
        costs = []
        actions_copy = actions.copy()
        play_set = []
        final_actions = []
        for action in actions_copy:
            if action['type'] == "place":
                if (action["play_card"],action["coords"]) in play_set:

                    continue
                play_set.append((action["play_card"],action["coords"]))
            elif action['type'] == "trade":
                final_actions.append(action)
                continue
            h = self.PlayAction(action)
            costs.append((action,h))
            total+=h

        costs = sorted(costs,key=lambda x: x[1])
        thresh = 0.1
        total_n = len(play_set)
        
        if total == 0:
            return actions_copy
        for action in costs:
            val = action[1]
            if val/total <= thresh:
                final_actions.append(action[0])
            else:
                break;
            
        pruned_n = total_n-len(final_actions)
        if len(final_actions)==0:
            print("no playing cards left. increase thresh")
            return [costs[-1][0]]

        print("PrunePlayActions pruned {} actions".format(str(pruned_n)))
        return final_actions

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
        else:
            h = 0
        print("new heuristic: ",h)
        return h
        

    def PlayAction(self,action,heuristic = "weighted"):
        """
        @params:
        actions = 1 action in dictionary format
        heuristic = string denoting which heuristic to use


        Takes 1 action dict and returns heuristic value for the
        play card only. 
        """ 
        play_coord = action['coords']
        action_type = action['type']
        if action_type == "trade":
            return 1
        if action_type == "place":
            ## These should be checked before entering into search ##
            try:
                if self.CheckNearWin(play_coord):
                    # IF 1 TILE AWAY FROM WINNING, RETURN 0 HEURISTIC                
                    return 0
            except:
                print("CheckNearWin Error for ",play_coord)
            try:
                if self.CheckNearLoss(play_coord):
                    # IF 1 TILE AWAY FROM LOSING, RETURN 0 HEURISTIC     
                    return 0
            except:
                print("CheckNearLoss Error for ",play_coord)
            try:
                if self.CheckHearts(play_coord):
                    # IF CAN PLAY HEARTS STRATEGY, RETURN 0 HEURISTIC
                    return 0
            except:
                print("CheckHearts Error for ",play_coord)
            d = self.CheckCurrentValue(self.goalstates,play_coord)

            if d:
                # COMPUTE HEURISTIC VALUE
                if heuristic == "weighted":
                    # print("weighted")
                    h = self.ComputeWeightedHeuristic(d)
                if heuristic == "newweighted":
                    print("newweighted")
                    h = self.ComputeNewWeightedHeuristic(d)
                elif heuristic == "simple":
                    # print("simple")
                    h = self.ComputeSimpleHeuristic(d)
                elif heuristic == "greedy":
                    print("greedy")
                    h = self.ComputeGreedyHeuristic(d)
                elif heuristic == "offensive":
                    print("offensive")
                    h = self.ComputeOffensiveHeuristic(d)
                elif heuristic == "defensive":
                    print("defensive")
                    h = self.ComputeDefensiveHeuristic(d)
            else:
                h = 10000
            return h
        elif action_type == "remove":
            # print("trying remove")
            if self.CheckNearLoss(play_coord):
                # print("removing near loss")
                return 0
        return 10000
       
    def DraftAction(self,action,heuristic = "weighted"):
        """
        @params:
        actions = 1 action in dictionary format
        heuristic = string denoting which heuristic to use

        Takes 1 action dict and returns h value for the draft card only.
        Using the PlayAction() function, it will play the action for each draft
        card and return a heuristic value for the current state of the board
        to determine which draft card is most valuable.
        """ 
        draft_card = action['draft_card']
        if draft_card == None:
            return 10000
        if draft_card in ['jd', 'jc','jh', 'js']:
            return 0  
 
        draft_coord = COORDS[draft_card] 
        if draft_coord in [(4,4),(5,4),(4,5),(4,5)]:
            return 0 
        total_h = []
        for coords in draft_coord:
            draft_action = {'play_card': draft_card, 
                            'draft_card': draft_card, 
                            'type': 'place', 
                            'coords': coords}
            h = self.PlayAction(draft_action, heuristic)
            total_h.append(h)
        return max(total_h)


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

        near_wins = [x for x in self.goalstates if len(x)==4 and x.count(1)==3]+[goal for goal in self.goalstates if (2<goal.count(1)<5 and goal.count(2)==1)]
        if near_wins:
            for goal in near_wins:
                for coord_goal in self.GetGoalsfromCoords(coord):
                    if coord_goal == goal:
                        return True
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

                d['sequence_count']+= 1 # Total number of sequences this coordinate exists in
                d['exclusive_goal_count'] += 1 if goal.count(2)==0 else 0 # Total potential goals where there are no opponent chips
                d['exclusive_player_tiles'] += math.e**goal.count(1) if goal.count(2)==0 else 0 # Total player tiles in our sequences without opponent tiles
                d['exclusive_opponent_tiles'] += math.e**goal.count(2) if goal.count(1)==0 else 0 
        return d
    
    # ======= COMPUTE HEURISTIC FUNCTIONS ======= #

    def ComputeNearWinHeuristic(self,d):
        """
        NOTE: This will return the opposite of a heuristic i.e. 0 is bad, 1 is good.
        Designed for QL
        """
        exclusive_player_tiles,exclusive_goal_count = d['exclusive_player_tiles'],d['exclusive_goal_count']
        return (0.5*exclusive_player_tiles/(exclusive_goal_count+1))

    def ComputeSimpleHeuristic(self,d):
        # Just the goals to win for each coord
        player_tiles,exclusive_goal_count = d['player_tiles'],d['exclusive_goal_count']
        return 1/player_tiles
    
    def ComputeNewWeightedHeuristic(self,d):
        player_tiles,opponent_tiles,sequence_count,exclusive_opponent_count = d['player_tiles'],d['opponent_tiles'],d['sequence_count'],d['exclusive_opponent_tiles']
        w1,w2,w3 = 0.5,0.3,0.2
        return(100/(w1*player_tiles)+100/(w3*sequence_count)+100/(w2*opponent_tiles+w2*exclusive_opponent_count))

    def ComputeWeightedHeuristic(self,d):
        player_tiles,opponent_tiles,sequence_count = d['player_tiles'],d['opponent_tiles'],d['sequence_count']
        w1,w2,w3 = 0.45,0.25,0.3
        return (100/(w1*player_tiles+w2*opponent_tiles+w3*sequence_count))

    def ComputeNewWeightedHeuristic(self,d):
        player_tiles,opponent_tiles,sequence_count = d['player_tiles'],d['opponent_tiles'],d['sequence_count']
        w1,w2,w3 = 0.45,0.25,0.3
        return (100/(w1*player_tiles+w2*opponent_tiles+w3*sequence_count))

    def ComputeGreedyHeuristic(self, d):
        w1,w2,w3 = 0.5,0.3,0.2
        easy_wins = d['exclusive_goal_count']+d['exclusive_player_tiles']
        potential_wins = d['player_tiles']
        easy_loss = (d['opponent_tiles']+d['exclusive_opponent_tiles'])/d['sequence_count']
        h = 1000/(w1*easy_wins+w2*potential_wins)+w3*easy_loss
        return h

    def ComputeOffensiveHeuristic(self,d):
        w1,w2 = 0.4,0.1
        easy_wins = d['exclusive_goal_count']/(d['exclusive_player_tiles']+1)
        potential_wins = d['sequence_count']/d['player_tiles']
        
        h = (10000/(w1*easy_wins+w2*potential_wins))
        return h
    
    def ComputeDefensiveHeuristic(self,d):
        w1,w2 = 0.4,0.1
        ours = d['exclusive_player_tiles']
        opp_tiles = d['exclusive_opponent_tiles']
        total = d['sequence_count']
        
        h = (10000/(w1*(opp_tiles+total)+w2*ours))
        return h

    def CardsToCoords(self,cards):
        hand_coords = []
        for card in cards:
            hand_coords= hand_coords+COORDS[card]
        return hand_coords

    def CoordsToCards(self,coords):
        x,y = coords
        return BOARD[x][y]

    def GetGoalsfromCoords(self,coords):
        goal_lst = []
        empty_board = self.InitiateBoard()
        for goal in empty_board:
            print("GET GOALS FROM COORDS: ",goal, " and  ", coords)
            if coords in goal:
                idx = empty_board.index(goal)
                goal_lst.append(self.goalstates[idx])
        return goal_lst
