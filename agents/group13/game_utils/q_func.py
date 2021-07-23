def AverageQfunc(node):
    """
    @params:
    node = the child of the root node, will be opponent's turn

    V value will be the average reward
    """
    if node.state.N == 0:
        return 0
    player_id = node.state.player_id
    # child player_id 1,3
    if player_id % 2:
        q = (node.state.win_scores_sum[0] + node.state.win_scores_sum[2]) / node.state.N
    # child player 0, 2
    else:
        q = (node.state.win_scores_sum[1] + node.state.win_scores_sum[3]) / node.state.N
    return q


def AggressiveQfunc(node):
    """
    @params:
    node = the child of the root node, will be opponent's turn

    V value will be the self reward minus the opponent reward
    """
    if node.state.N == 0:
        return 0
    player_id = node.state.player_id
    # child player_id 1,3
    if player_id % 2:
        q = (node.state.win_scores_sum[0] + node.state.win_scores_sum[2]
             - node.state.win_scores_sum[1] - node.state.win_scores_sum[3]) / node.state.N
    # child player 0, 2
    else:
        q = (node.state.win_scores_sum[1] + node.state.win_scores_sum[3]
             - node.state.win_scores_sum[0] - node.state.win_scores_sum[2]) / node.state.N
    return q
