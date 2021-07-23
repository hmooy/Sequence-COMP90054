import math



def eucDist(action, pCoords):
    """
    Euclidean distance centroid feature
   """
    centr = centroid(pCoords)

    if action['type'] == 'place':
        card_coord = action['coords']
        eucDistCentroid = math.sqrt((centr[0] - card_coord[0])**2 + (centr[1] - card_coord[1])**2)
        return eucDistCentroid
    else:
        return 0.0


def centroid(list_coords):
    """
    Return the centroid given the list of coords
    """
    x_coords = [i[0] for i in list_coords]
    y_coords = [j[1] for j in list_coords]
    _len = len(list_coords)
    if _len == 0:
        return (0,0)
    else:
        x_centroid = sum(x_coords)/_len
        y_centroid = sum(y_coords)/_len
    return (x_centroid, y_centroid)


def neighbour(action, plrCoords, oppCoords):
    '''
    Check the 8cells around action
    return the number of alias - number of opponent
    normalize with total number of adjacent cells
    '''

    if action['type']=='place':
        xCoords,yCoords = action['coords']
        numberNeighbour = 0
        totalLegalCell = 0
        for x, y in [(xCoords+i, yCoords+j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i != 0 or j != 0]:
            if 0 < x < 10 and 0 < y < 10:
                totalLegalCell += 1
            if (x, y) in plrCoords:
                numberNeighbour += 1
            if (x, y) in oppCoords:
                numberNeighbour -= 1
        return numberNeighbour/totalLegalCell

    else:
        return 0.0


def heart(action, plrCoords):
    '''
    Centre of the board (2, 3, 4, and 5) are the most valuable spots in the game
    '''
    centreCoords = [(4,4),(5,4),(4,5),(5,5)]
    occupied = 0
    for x in centreCoords:
        if x in plrCoords:
            occupied += 1
    if action['type'] == 'place' and action['coords'] in centreCoords:
        return (occupied + 1)/4
    else:
        return 0.0


def blockHeart(action, oppCoords):
    '''
    Centre of the board (2, 3, 4, and 5) are the most valuable spots in the game
    If opponent occupied one space will give 0.25
    1-total occupy, therefore less occupied closer to 1
    '''

    centreCoords = [(4,4),(5,4),(4,5),(5,5)]
    occupied = 0
    for x in centreCoords:
        if x in oppCoords:
            occupied += 0.25
    if occupied > 0 and action['type'] == 'place' and action['coords'] in centreCoords:
        return 1-(occupied-0.25)
    else:
        return 0.0


def eHorizontal(state, action, plrCoords, oppCoords):
    '''
    Expectation of getting a sequence
    on that horizontal row, if greater than 5 occupied
    expected value is negative
    '''

    if action['type'] == 'place':
        row, _ = action['coords']
        empCoords = state.board.empty_coords
        myChips = 0
        emptyCell = 0
        oppChips = 0
        for i in plrCoords:
            if i[0] == row:
                myChips += 1
        for j in oppCoords:
            if j[0] == row:
                oppChips += 1
        for k in empCoords:
            if k[0] == row:
                emptyCell += 1

        if row == 0 or 9:
            prob = (myChips+emptyCell-oppChips)/8
        else:
            prob = (myChips+emptyCell-oppChips)/10

        return prob
    else:
        return 0.0


def eVertical(state, action, plrCoords, oppCoords):
    '''
    Expectation of getting a sequence
    on that vertical row, if greater than 5 occupied
    expected value is negative
    '''

    if action['type'] == 'place':
        _, col = action['coords']
        empCoords = state.board.empty_coords
        myChips = 0
        emptyCell = 0
        oppChips = 0
        for i in plrCoords:
            if i[1] == col:
                myChips += 1
        for j in oppCoords:
            if j[1] == col:
                oppChips += 1
        for k in empCoords:
            if k[1] == col:
                emptyCell += 1

        if col == 0 or 9:
            prob = (myChips+emptyCell-oppChips)/8
        else:
            prob = (myChips+emptyCell-oppChips)/10

        return prob
    else:
        return 0.0


def eIandIIIDiagonal(state, action, pCoords, oCoords):
    '''
    Expectation of getting a sequence
    on diagonal in quadrant I and III
    from action's coordinate
    '''
    if action['type'] == 'place':
        row, col = action['coords']
        listDiag = diagonal(row, col, "IandIII")
        empCoords = state.board.empty_coords
        myChips = len(set(listDiag).intersection(set(pCoords)))
        emptyCell = len(set(listDiag).intersection(set(oCoords)))
        oppChips = len(set(listDiag).intersection(set(empCoords)))

        if (0, 9) in listDiag:
            return (myChips+emptyCell-oppChips)/(len(listDiag)+1-2)
        else:
            return (myChips+emptyCell-oppChips)/(len(listDiag)+1)
    else:
        return 0.0


def eIIandIVDiagonal(state, action, pCoords, oCoords):
    '''
    Expectation of getting a sequence
    on diagonal in quadrant I and III
    from action's coordinate
    '''
    if action['type'] == 'place':
        row, col = action['coords']
        listDiag = diagonal(row, col, "IIandIV")
        empCoords = state.board.empty_coords
        myChips = len(set(listDiag).intersection(set(pCoords)))
        emptyCell = len(set(listDiag).intersection(set(oCoords)))
        oppChips = len(set(listDiag).intersection(set(empCoords)))

        if (0, 0) in listDiag:
            return (myChips+emptyCell-oppChips)/(len(listDiag)+1-2)
        else:
            return (myChips+emptyCell-oppChips)/(len(listDiag)+1)
    else:
        return 0.0


def diagonal(row, col, angle="IandIII"):
    """
    Given a (row,col) coordinate return all diagonal coordinates
    """
    coords = []
    rowc = row
    colc = col
    if angle == "IandIII":
        while 0 < row and col < 9:
            row -= 1
            col += 1
            coords.append((row, col))
        while rowc < 9 and 0 < colc:
            rowc += 1
            colc -= 1
            coords.append((rowc, colc))

    if angle == "IIandIV":
        while row < 9 and col < 9:
            row += 1
            col += 1
            coords.append((row,col))

        while rowc > 0 and colc > 0:
            rowc -= 1
            colc -= 1
            coords.append((rowc, colc))

    return coords


def draftHorizontal(state, plrCoords, oppCoords, draftCoords):
    '''
    Draft card that could be useful to agent
    based on the average horizontal values of the two coords
    obtained from draft card coordinates
    '''
    empCoords = state.board.empty_coords
    value = 0
    if draftCoords == None:
        return 0
    else:
        for coord in draftCoords:
            row, _ = coord
            myChips = 0
            emptyCell = 0
            oppChips = 0
            for i in plrCoords:
                if i[0] == row:
                    myChips += 1
            for j in oppCoords:
                if j[0] == row:
                    oppChips += 1
            for k in empCoords:
                if k[0] == row:
                    emptyCell += 1

            if row == 0 or 9:
                value += (myChips+emptyCell-oppChips)/8
            else:
                value += (myChips+emptyCell-oppChips)/10

        average = value/2

    return average


def draftVertical(state, plrCoords, oppCoords, draftCoords):
    '''
    Draft card that could be useful to agent
    based on the average vertical values of the two coords
    obtained from two draft cards coordinates
    '''
    empCoords = state.board.empty_coords
    value = 0
    if draftCoords == None:
        return 0
    else:
        for coord in draftCoords:
            _, col = coord

            myChips = 0
            emptyCell = 0
            oppChips = 0
            for i in plrCoords:
                if i[1] == col:
                    myChips += 1
            for j in oppCoords:
                if j[1] == col:
                    oppChips += 1
            for k in empCoords:
                if k[1] == col:
                    emptyCell += 1

            if col == 0 or 9:
                value += (myChips+emptyCell-oppChips)/8
            else:
                value += (myChips+emptyCell-oppChips)/10

        average = value/2

    return average


def draftDiagIandIII(state, pCoords, oCoords, draftCoords):
    '''
    Draft card that could be useful to agent
    based on the average diagonal values of the two coords
    obtained from two draft cards coordinates
    '''
    empCoords = state.board.empty_coords
    value = 0
    if draftCoords == None:
        return 0
    else:
        for coord in draftCoords:
            row, col = coord
            listDiag = diagonal(row, col, "IandIII")
            myChips = len(set(listDiag).intersection(set(pCoords)))
            emptyCell = len(set(listDiag).intersection(set(oCoords)))
            oppChips = len(set(listDiag).intersection(set(empCoords)))

            if (0, 9) in listDiag:
                value += (myChips+emptyCell-oppChips)/(len(listDiag)+1-2)
            else:
                value += (myChips+emptyCell-oppChips)/(len(listDiag)+1)
        average = value/2

    return average


def draftDiagIIandIV(state, pCoords, oCoords, draftCoords):
    '''
    Draft card that could be useful to agent
    based on the average diagonal values of the two coords
    obtained from two draft cards coordinates
    '''
    empCoords = state.board.empty_coords
    value = 0
    if draftCoords == None:
        return 0
    else:
        for coord in draftCoords:
            row, col = coord
            listDiag = diagonal(row, col, "IIandIV")
            myChips = len(set(listDiag).intersection(set(pCoords)))
            emptyCell = len(set(listDiag).intersection(set(oCoords)))
            oppChips = len(set(listDiag).intersection(set(empCoords)))

            if (0, 0) in listDiag:
                value += (myChips+emptyCell-oppChips)/(len(listDiag)+1-2)
            else:
                value += (myChips+emptyCell-oppChips)/(len(listDiag)+1)
        average = value/2

    return average


def DraftJacks(action):
    '''
    Draft jacks card that could be useful to agent
    '''
    if action['draft_card'] in ['jd', 'jc']:
        return 1.0
    elif action['draft_card'] in ['js', 'jh']:
        return 0.5
    else:
        return 0.0


def PlayCentre(action):
    '''
    Play centre of the board
    '''
    if action['coords'] is not None:
        centre = (4.5, 4.5)
        card_coord = action['coords']
        eucDistCentroid = math.sqrt((centre[0] - card_coord[0])**2 + (centre[1] - card_coord[1])**2)
        maxDist = math.sqrt((centre[0] - 0)**2 + (centre[1] - 9)**2)
        normEucDist = eucDistCentroid/maxDist

        return normEucDist
    else:
        return 0.0


def HeuristicValue(action, goalState):
    '''
    Compute heuristic value based on strategy
    Action that has high heuristic value is good
    '''
    if action['coords'] is not None and action['type'] == 'place':
        h = goalState.QlearningAction(action['coords'])
        return h
    else:
        return 0.0


def HeuristicValueDraft(action, goalState, draftCoords, gamma):
    '''
    Compute heuristic value based on strategy
    Action that has high heuristic value is good
    '''
    h = 0.0
    if draftCoords is not None:
        for coords in draftCoords:
            h += goalState.QlearningAction(coords)
    return gamma*h/2


