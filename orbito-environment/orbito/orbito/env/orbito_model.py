import numpy as np

agents = ["player_0", "player_1"]
possible_agents = agents[:]

def _legal_moves(state):
    board = to1by16(state)
    return [i for i in range(16) if board[i] == 0]

# the available actions for the current player to move the opponents piece horizontally or vertically by one space
def _legal_manipulate_pieces(state, opp_p_board):
    board = to1by16(state)
    positions = [i for i in range(16) if opp_p_board.flatten()[i]]
    legal_moves = [[] for x in range(16)]
    for i in positions:
        if i % 4 != 0 and board[i-1] == 0:
            legal_moves[i].append(i-1)
        if i % 4 != 3 and board[i+1] == 0:
            legal_moves[i].append(i+1)
        if i > 3 and board[i-4] == 0:
            legal_moves[i].append(i-4)
        if i < 12 and board[i+4] == 0:
            legal_moves[i].append(i+4)
    return legal_moves

def rotatePieces(state):
    new_board = state.copy()
    for i in [1,2,3]:
        new_board[i-1] = state[i]
    for i in [0,4,8]:
        new_board[i+4] = state[i]
    for i in [12,13,14]:
        new_board[i+1] = state[i]
    for i in [15,11,7]:
        new_board[i-4] = state[i]
    new_board[5] = state[6]
    new_board[9] = state[5]
    new_board[10] = state[9]
    new_board[6] = state[10]
    return new_board

def to4by4(state):
    return [state[:4], state[4:8], state[8:12], state[12:16]]

def to1by16(state):
    return [item for sublist in state for item in sublist]

class OrbitoModel:


    def ACTIONS(state, agent):
        board_vals = state.copy()
        cur_player = possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        cur_p_board = np.equal(board_vals, cur_player + 1)
        opp_p_board = np.equal(board_vals, opp_player + 1)
                                                                                                                #fix agent selection
        # observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
        observation = board_vals
        legal_manipulate_pieces = _legal_manipulate_pieces(state, opp_p_board.astype(np.int8)) # if agent == agent_selection else [[] for i in range(16)]
        legal_moves = _legal_moves(state) # if agent == agent_selection else []
        # manipulate_mask = [np.zeros(16, "int8") for _ in range(16)]
        # for i in range(len(legal_manipulate_pieces)):
        #     for j in legal_manipulate_pieces[i]:
        #         manipulate_mask[i][j] = 1
        action_mask = np.zeros(16, "int8")
        for i in legal_moves:
            action_mask[i] = 1
        return {"observation": observation, "action_mask": action_mask, "manipulate_mask": legal_manipulate_pieces}

    def human_RESULT_1(state, action_data, agent):
        board = to1by16(state.copy())
        # action, manipulate_piece, manipulate_action = self.decode_values(action_data)
        action, manipulate_piece, manipulate_action, manipulate = (action_data[0], action_data[1], action_data[2], action_data[3])
        next_agent = 'player_0' if agent == 'player_1' else 'player_1'
        opp_piece = agents.index(next_agent) + 1
        if manipulate:
            # assert valid manipulate move
            assert board[0:16][manipulate_piece] == opp_piece, "played illegal move."
            assert board[0:16][manipulate_action] == 0, "played illegal move."
            board[manipulate_piece] = 0
            board[manipulate_action] = opp_piece
        
        return to4by4(board)
    
    def human_RESULT_2(state, action_data, agent):
        board = to1by16(state.copy())
        # action, manipulate_piece, manipulate_action = self.decode_values(action_data)
        action, manipulate_piece, manipulate_action, manipulate = (action_data[0], action_data[1], action_data[2], action_data[3])
                # assert valid move
        assert board[0:16][action] == 0, "played illegal move."

        piece = agents.index(agent) + 1
        for i in list(filter(lambda x: x % 16 == action, list(range(15, -1, -1)))):
            if board[i] == 0:
                board[i] = piece
                break
        return to4by4(board)
    
    def human_RESULT_3(state):
        board = to1by16(state.copy())
        rotated_board = rotatePieces(board)
        return to4by4(rotated_board)

    def RESULT(state, action, agent):
        board = to1by16(state.copy())
        # action, manipulate_piece, manipulate_action = self.decode_values(action_data)
        action, manipulate_piece, manipulate_action, manipulate = (action[0], action[1], action[2], action[3])
        next_agent = 'player_0' if agent == 'player_1' else 'player_1'
        opp_piece = agents.index(next_agent) + 1
        if manipulate:
            # assert valid manipulate move
            assert board[0:16][manipulate_piece] == opp_piece, "played illegal move."
            assert board[0:16][manipulate_action] == 0, "played illegal move."
            board[manipulate_piece] = 0
            board[manipulate_action] = opp_piece
        # assert valid move
        assert board[0:16][action] == 0, "played illegal move."

        piece = agents.index(agent) + 1
        for i in list(filter(lambda x: x % 16 == action, list(range(15, -1, -1)))):
            if board[i] == 0:
                board[i] = piece
                break
        #render the move played then rotate the board
        rotated_board = rotatePieces(board)
        return to4by4(rotated_board)

    def GOAL_TEST(state):
        board = state.copy()
        winner = []
        for piece in range(len(agents)):
            piece += 1
            # Check horizontal locations for win
            column_count = 4
            row_count = 4

            for c in range(column_count - 3):
                for r in range(row_count):
                    if (
                        board[r][c] == piece
                        and board[r][c + 1] == piece
                        and board[r][c + 2] == piece
                        and board[r][c + 3] == piece
                    ):
                        winner.append(piece-1)

            # Check vertical locations for win
            for c in range(column_count):
                for r in range(row_count - 3):
                    if (
                        board[r][c] == piece
                        and board[r + 1][c] == piece
                        and board[r + 2][c] == piece
                        and board[r + 3][c] == piece
                    ):
                        if piece-1 not in winner:
                            winner.append(piece-1)
                        

            # Check positively sloped diagonals
            for c in range(column_count - 3):
                for r in range(row_count - 3):
                    if (
                        board[r][c] == piece
                        and board[r + 1][c + 1] == piece
                        and board[r + 2][c + 2] == piece
                        and board[r + 3][c + 3] == piece
                    ):
                        if piece-1 not in winner:
                            winner.append(piece-1)

            # Check negatively sloped diagonals
            for c in range(column_count - 3):
                for r in range(3, row_count):
                    if (
                        board[r][c] == piece
                        and board[r - 1][c + 1] == piece
                        and board[r - 2][c + 2] == piece
                        and board[r - 3][c + 3] == piece
                    ):
                        if piece-1 not in winner:
                            winner.append(piece-1)

        return winner

    def STEP_COST(state, action):
        pass

    def HEURISTIC(state):
        pass