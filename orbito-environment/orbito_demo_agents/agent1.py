import random
import numpy as np
from orbito.env.orbito_model import OrbitoModel, rotatePieces
INFINITY = 1.e10

# itterative deepening and shallow search... update move ordering

class OrbitoAgent:

    def __init__(self):
        self.cutoff_depth = 4
        self.players = ['player_0', 'player_1']
        self.state_action_dict = {}
        return
    
    def reset(self):
        return

    

    def MINIMAX(self, observation, agent):
        self.player = agent
        player = agent
        best_value = -INFINITY if player else INFINITY
        best_action = None

        # Move Ordering
        ordered_actions = self.move_ordering(observation, player)

        for action in ordered_actions:
            state_key = (tuple(map(tuple, observation['observation'])), tuple(action))
            if state_key in self.state_action_dict:
                value = self.state_action_dict[state_key]
            else:
                result = OrbitoModel.RESULT(observation['observation'], action, player)
                next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                value = self.MIN(next_state, 1, -INFINITY, INFINITY, self.players[0] if player != self.players[0] else self.players[1])
                self.state_action_dict[state_key] = value

            if (player and value > best_value) or (not player and value < best_value):
                best_value = value
                best_action = action

                # Early Termination
                if (player and value == INFINITY) or (not player and value == -INFINITY):
                    return best_action

        return best_action

    def MAX(self, current_state, depth, alpha, beta, player):
        if depth >= self.cutoff_depth or self.GAME_OVER(current_state['observation']):
            return self.EVALUATE(current_state)

        best_value = -INFINITY

        # Move Ordering
        ordered_actions = self.move_ordering(current_state, player)

        # Shallow Search
        shallow_depth = 2
        if depth < shallow_depth:
            ordered_actions = ordered_actions[:len(ordered_actions)//2]

        # Alpha-Beta Pruning
        for action in ordered_actions:
            state_key = (tuple(map(tuple, current_state['observation'])), tuple(action))
            if state_key in self.state_action_dict:
                value = self.state_action_dict[state_key]
            else:
                result = OrbitoModel.RESULT(current_state['observation'], action, player)
                next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                value = self.MIN(next_state, depth + 1, alpha, beta, self.players[0] if player != self.players[0] else self.players[1])
                self.state_action_dict[state_key] = value

            if value > best_value:
                best_value = value
                alpha = max(alpha, best_value)

            if beta <= alpha:
                break  # Pruning

        return best_value

    def MIN(self, current_state, depth, alpha, beta, player):
        if depth >= self.cutoff_depth or self.GAME_OVER(current_state['observation']):
            return self.EVALUATE(current_state)

        best_value = INFINITY

        # Move Ordering
        ordered_actions = self.move_ordering(current_state, player)

        # Shallow Search
        shallow_depth = 2
        if depth < shallow_depth:
            ordered_actions = ordered_actions[:len(ordered_actions)//2]

        # Alpha-Beta Pruning
        for action in ordered_actions:
            state_key = (tuple(map(tuple, current_state['observation'])), tuple(action))
            if state_key in self.state_action_dict:
                value = self.state_action_dict[state_key]
            else:
                result = OrbitoModel.RESULT(current_state['observation'], action, player)
                next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                value = self.MAX(next_state, depth + 1, alpha, beta, self.players[0] if player != self.players[0] else self.players[1])
                self.state_action_dict[state_key] = value

            if value < best_value:
                best_value = value
                beta = min(beta, best_value)

            if beta <= alpha:
                break  # Pruning

        return best_value

    
    def move_ordering(self, state, player):
        actions = []
        manipulate_choices = [i for i, mask in enumerate(state['manipulate_mask']) if mask]
        
        for i in manipulate_choices:
            for j in state['manipulate_mask'][i]:
                possible_actions = np.setdiff1d(np.where(state['action_mask'])[0], j)
                possible_actions = np.append(possible_actions, i)
                actions.extend([(k, i, j, 1) for k in possible_actions])
        
        actions.extend([(i, 0, 0, 0) for i in np.where(state['action_mask'])[0]])
        
        # Implement a heuristic to prioritize moves
        # For example, prioritize moves that result in immediate wins or blocks
        def heuristic(action):
            result = OrbitoModel.RESULT(state['observation'], action, player)
            score = 0
            if self.GAME_OVER(result):
                eval = self.EVALUATE({'observation': result})
                return INFINITY if eval == 1 else -INFINITY if eval == -1 else 0
            heuristics = self.gather_heuristics(result, player)
            if heuristics['three_in_a_row']:
                score += 6
            elif heuristics['two_in_a_row']:
                score += 4
            if heuristics['enemy_three_in_a_row']:
                score -= 10
            if heuristics['block_enemy_two_in_a_row']:
                score += 3
            # elif self.enemy_three_in_a_row(result, player):
            #     # print('three in a row')
            #     score -= 10
            if action[0] in [5,6,9,10]:
                score += 2
            # elif self.two_in_a_row(result, player):
            #     score+= 5  # Positive score for having two in a row
            # elif self.block_enemy_two_in_a_row(result, player):
            #     score+= 3  # Positive score for blocking enemy's two in a row
            return score
        
        actions.sort(key=heuristic, reverse=True)
        return actions

    def gather_heuristics(self, observation, player):
        enemy = 'player_0' if player == 'player_1' else 'player_1'
        enemyNum = 1 if enemy == 'player_0' else 2
        pNum = 1 if player == 'player_0' else 2
        heuristics = {'three_in_a_row': False, 'two_in_a_row': False, 'block_enemy_two_in_a_row': False, 'enemy_three_in_a_row': False}
        lines = [
            [observation[0][1], observation[0][0], observation[1][0], observation[2][0]],
            [observation[0][2], observation[1][2], observation[1][1], observation[3][0]],
            [observation[0][3], observation[2][2], observation[2][1], observation[3][1]],
            [observation[1][3], observation[2][3], observation[3][3], observation[3][2]],
            [observation[0][1], observation[1][2], observation[2][1], observation[3][2]],
            [observation[1][3], observation[1][1], observation[2][2], observation[2][0]],
            [observation[0][1], observation[0][2], observation[0][3], observation[1][3]],
            [observation[0][0], observation[1][2], observation[2][2], observation[2][3]],
            [observation[1][0], observation[1][1], observation[2][1], observation[3][3]],
            [observation[2][0], observation[3][0], observation[3][1], observation[3][2]]
        ]
        for line in lines:
            if sum(1 for j in line if j == pNum) == 3:
                heuristics['three_in_a_row'] = True
            if sum(1 for j in line if j == pNum) == 2 and sum(1 for j in line if j == 0) >= 1:
                heuristics['two_in_a_row'] = True
            if sum(1 for j in line if j == enemyNum) == 2 and sum(1 for j in line if j != enemyNum) >= 2:
                heuristics['block_enemy_two_in_a_row'] = True
            if sum(1 for j in line if j == enemyNum) == 3:
                heuristics['enemy_three_in_a_row'] = True
        return heuristics

    def enemy_three_in_a_row(self, observation, player):
        enemy = 'player_0' if player == 'player_1' else 'player_1'
        pNum = 1 if enemy == 'player_0' else 2
        #verticals
        lines = [
            [observation[0][1], observation[0][0], observation[1][0], observation[2][0]],
            [observation[0][2], observation[1][2], observation[1][1], observation[3][0]],
            [observation[0][3], observation[2][2], observation[2][1], observation[3][1]],
            [observation[1][3], observation[2][3], observation[3][3], observation[3][2]],
            [observation[0][1], observation[1][2], observation[2][1], observation[3][2]],
            [observation[1][3], observation[1][1], observation[2][2], observation[2][0]],
            [observation[0][1], observation[0][2], observation[0][3], observation[1][3]],
            [observation[0][0], observation[1][2], observation[2][2], observation[2][3]],
            [observation[1][0], observation[1][1], observation[2][1], observation[3][3]],
            [observation[2][0], observation[3][0], observation[3][1], observation[3][2]]
        ]
        
        for line in lines:
            if sum(1 for j in line if j == pNum) >= 3:
                return True
        return False
        

    def two_in_a_row(self, observation, player):
        pNum = 1 if player == 'player_0' else 2
        lines = [
            [observation[0][1], observation[0][0], observation[1][0], observation[2][0]],
            [observation[0][2], observation[1][2], observation[1][1], observation[3][0]],
            [observation[0][3], observation[2][2], observation[2][1], observation[3][1]],
            [observation[1][3], observation[2][3], observation[3][3], observation[3][2]],
            [observation[0][1], observation[1][2], observation[2][1], observation[3][2]],
            [observation[1][3], observation[1][1], observation[2][2], observation[2][0]],
            [observation[0][1], observation[0][2], observation[0][3], observation[1][3]],
            [observation[0][0], observation[1][2], observation[2][2], observation[2][3]],
            [observation[1][0], observation[1][1], observation[2][1], observation[3][3]],
            [observation[2][0], observation[3][0], observation[3][1], observation[3][2]]
        ]
        
        for line in lines:
            if sum(1 for j in line if j == pNum) == 2 and sum(1 for j in line if j == 0) >= 1:
                return True
        return False

    def block_enemy_two_in_a_row(self, observation, player):
        enemy = 'player_0' if player == 'player_1' else 'player_1'
        pNum = 1 if enemy == 'player_0' else 2
        lines = [
            [observation[0][1], observation[0][0], observation[1][0], observation[2][0]],
            [observation[0][2], observation[1][2], observation[1][1], observation[3][0]],
            [observation[0][3], observation[2][2], observation[2][1], observation[3][1]],
            [observation[1][3], observation[2][3], observation[3][3], observation[3][2]],
            [observation[0][1], observation[1][2], observation[2][1], observation[3][2]],
            [observation[1][3], observation[1][1], observation[2][2], observation[2][0]],
            [observation[0][1], observation[0][2], observation[0][3], observation[1][3]],
            [observation[0][0], observation[1][2], observation[2][2], observation[2][3]],
            [observation[1][0], observation[1][1], observation[2][1], observation[3][3]],
            [observation[2][0], observation[3][0], observation[3][1], observation[3][2]]
        ]
        
        for line in lines:
            if sum(1 for j in line if j == pNum) == 2 and sum(1 for j in line if j != pNum) >= 2:
                return True
        return False
    
    
    
    def EVALUATE(self, state):
        result = OrbitoModel.GOAL_TEST(state['observation'])
        if len(result) > 0:
            if len(result) == 2:
                return 0
            if result[0] == 0 and self.player == 'player_0':
                return 1
            elif result[0] == 1 and self.player == 'player_1':
                return 1
            else:
                return -1

        return 0
    
    def GAME_OVER(self, state):
        result = OrbitoModel.GOAL_TEST(state)
        if len(result) > 0:
            return True
        return False

    def agent_function(self, observation, agent):
        action = self.MINIMAX(observation, agent)
        return action
    






class OrbitoAgentV1:

    def __init__(self):
        self.cutoff_depth = 2
        self.players = ['player_0', 'player_1']
        return
    
    def reset(self):
        return


    def MINIMAX(self, observation, agent):
        self.player = agent
        player = agent
        best_value = -INFINITY if player else INFINITY
        best_action = None

        manipulate_choices = []
        for i in range(len(observation['manipulate_mask'])):
            if observation['manipulate_mask'][i]:
                manipulate_choices.append(i)
        if len(manipulate_choices) > 0:
            for i in manipulate_choices:
                for j in observation['manipulate_mask'][i]:
                    possible_actions = np.where(observation['action_mask'])[0]
                    possible_actions = np.delete(possible_actions, np.where(possible_actions == j))
                    possible_actions = np.append(possible_actions, i)
                    for k in possible_actions:
                        result = OrbitoModel.RESULT(observation['observation'], [k, i, j, 1], player)
                        next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                        value = self.MIN(next_state, 1, self.players[0] if player != self.players[0] else self.players[1])
                        if (player and value > best_value) or (not player and value < best_value):
                            best_value = value
                            best_action = [k, i, j, 1]
        else:
            for i in observation['action_mask'].nonzero()[0]:
                result = OrbitoModel.RESULT(observation['observation'], [i, 0, 0, 0], player)
                next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                value = self.MIN(next_state, 1, self.players[0] if player != self.players[0] else self.players[1])
                if (value > best_value):
                    best_value = value
                    best_action = [i, 0, 0, 0]
        return best_action

    

    def MAX(self, current_state, depth, player):
        if depth >= self.cutoff_depth or self.GAME_OVER(current_state['observation']):
            return self.EVALUATE(current_state)
        best_value = -INFINITY
        manipulate_choices = []
        for i in range(len(current_state['manipulate_mask'])):
            if current_state['manipulate_mask'][i]:
                manipulate_choices.append(i)
        if len(manipulate_choices) > 0:
            for i in manipulate_choices:
                for j in current_state['manipulate_mask'][i]:
                    possible_actions = np.where(current_state['action_mask'])[0]
                    possible_actions = np.delete(possible_actions, np.where(possible_actions == j))
                    possible_actions = np.append(possible_actions, i)
                    for k in possible_actions:
                        result = OrbitoModel.RESULT(current_state['observation'], [k, i, j, 1], player)
                        next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                        value = self.MIN(next_state, depth+1, self.players[0] if player != self.players[0] else self.players[1])
                        if value > best_value:
                            best_value = value
        else:
            for i in current_state['action_mask'].nonzero()[0]:
                result = OrbitoModel.RESULT(current_state['observation'], [i, 0, 0, 0], player)
                next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                value = self.MIN(next_state, depth+1, self.players[0] if player != self.players[0] else self.players[1])
                if value > best_value:
                    best_value = value
        return best_value

    def MIN(self, current_state, depth, player):
        if depth >= self.cutoff_depth or self.GAME_OVER(current_state['observation']):
            return self.EVALUATE(current_state)
        best_value = INFINITY
        manipulate_choices = []
        for i in range(len(current_state['manipulate_mask'])):
            if current_state['manipulate_mask'][i]:
                manipulate_choices.append(i)
        if len(manipulate_choices) > 0:
            for i in manipulate_choices:
                for j in current_state['manipulate_mask'][i]:
                    possible_actions = np.where(current_state['action_mask'])[0]
                    possible_actions = np.delete(possible_actions, np.where(possible_actions == j))
                    possible_actions = np.append(possible_actions, i)
                    for k in possible_actions:
                        result = OrbitoModel.RESULT(current_state['observation'], [k, i, j, 1], player)
                        next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                        value = self.MAX(next_state, depth+1, self.players[0] if player != self.players[0] else self.players[1])
                        if value < best_value:
                            best_value = value
        else:
            for i in current_state['action_mask'].nonzero()[0]:
                result = OrbitoModel.RESULT(current_state['observation'], [i, 0, 0, 0], player)
                next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                value = self.MAX(next_state, depth+1, self.players[0] if player != self.players[0] else self.players[1])
                if value < best_value:
                    best_value = value
        return best_value

    def EVALUATE(self, state):
        result = OrbitoModel.GOAL_TEST(state['observation'])
        if len(result) > 0:
            if len(result) == 2:
                return 0
            if result[0] == 0 and self.player == 'player_0':
                return 1
            elif result[0] == 1 and self.player == 'player_1':
                return 1
            else:
                return -1

        return 0
    
    def GAME_OVER(self, state):
        result = OrbitoModel.GOAL_TEST(state)
        if len(result) > 0:
            return True
        return False


    def agent_function(self, observation, agent):
        action = self.MINIMAX(observation, agent)
        return action
    

class OrbitoAgentV2:

    def __init__(self):
        self.cutoff_depth = 4
        self.players = ['player_0', 'player_1']
        self.state_action_dict = {}
        return
    
    def reset(self):
        return

    

    def MINIMAX(self, observation, agent):
        self.player = agent
        player = agent
        best_value = -INFINITY if player else INFINITY
        best_action = None

        # Move Ordering
        ordered_actions = self.move_ordering(observation, player)

        for action in ordered_actions:
            state_key = (tuple(map(tuple, observation['observation'])), tuple(action))
            if state_key in self.state_action_dict:
                value = self.state_action_dict[state_key]
            else:
                result = OrbitoModel.RESULT(observation['observation'], action, player)
                next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                value = self.MIN(next_state, 1, -INFINITY, INFINITY, self.players[0] if player != self.players[0] else self.players[1])
                self.state_action_dict[state_key] = value

            if (player and value > best_value) or (not player and value < best_value):
                best_value = value
                best_action = action

                # Early Termination
                if (player and value == INFINITY) or (not player and value == -INFINITY):
                    return best_action

        return best_action

    def MAX(self, current_state, depth, alpha, beta, player):
        if depth >= self.cutoff_depth or self.GAME_OVER(current_state['observation']):
            return self.EVALUATE(current_state)

        best_value = -INFINITY

        # Move Ordering
        ordered_actions = self.move_ordering(current_state, player)

        # Shallow Search
        shallow_depth = 2
        if depth < shallow_depth:
            ordered_actions = ordered_actions[:len(ordered_actions)//2]

        # Alpha-Beta Pruning
        for action in ordered_actions:
            state_key = (tuple(map(tuple, current_state['observation'])), tuple(action))
            if state_key in self.state_action_dict:
                value = self.state_action_dict[state_key]
            else:
                result = OrbitoModel.RESULT(current_state['observation'], action, player)
                next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                value = self.MIN(next_state, depth + 1, alpha, beta, self.players[0] if player != self.players[0] else self.players[1])
                self.state_action_dict[state_key] = value

            if value > best_value:
                best_value = value
                alpha = max(alpha, best_value)

            if beta <= alpha:
                break  # Pruning

        return best_value

    def MIN(self, current_state, depth, alpha, beta, player):
        if depth >= self.cutoff_depth or self.GAME_OVER(current_state['observation']):
            return self.EVALUATE(current_state)

        best_value = INFINITY

        # Move Ordering
        ordered_actions = self.move_ordering(current_state, player)

        # Shallow Search
        shallow_depth = 2
        if depth < shallow_depth:
            ordered_actions = ordered_actions[:len(ordered_actions)//2]

        # Alpha-Beta Pruning
        for action in ordered_actions:
            state_key = (tuple(map(tuple, current_state['observation'])), tuple(action))
            if state_key in self.state_action_dict:
                value = self.state_action_dict[state_key]
            else:
                result = OrbitoModel.RESULT(current_state['observation'], action, player)
                next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
                value = self.MAX(next_state, depth + 1, alpha, beta, self.players[0] if player != self.players[0] else self.players[1])
                self.state_action_dict[state_key] = value

            if value < best_value:
                best_value = value
                beta = min(beta, best_value)

            if beta <= alpha:
                break  # Pruning

        return best_value

    
    def move_ordering(self, state, player):
        actions = []
        manipulate_choices = [i for i, mask in enumerate(state['manipulate_mask']) if mask]
        
        for i in manipulate_choices:
            for j in state['manipulate_mask'][i]:
                possible_actions = np.setdiff1d(np.where(state['action_mask'])[0], j)
                possible_actions = np.append(possible_actions, i)
                actions.extend([(k, i, j, 1) for k in possible_actions])
        
        actions.extend([(i, 0, 0, 0) for i in np.where(state['action_mask'])[0]])
        
        # Implement a heuristic to prioritize moves
        # For example, prioritize moves that result in immediate wins or blocks
        def heuristic(action):
            result = OrbitoModel.RESULT(state['observation'], action, player)
            if self.GAME_OVER(result):
                return INFINITY if self.EVALUATE({'observation': result}) == 1 else -INFINITY
            return 0
        
        actions.sort(key=heuristic, reverse=True)
        return actions
    
    
    
    def EVALUATE(self, state):
        result = OrbitoModel.GOAL_TEST(state['observation'])
        if len(result) > 0:
            if len(result) == 2:
                return 0
            if result[0] == 0 and self.player == 'player_0':
                return 1
            elif result[0] == 1 and self.player == 'player_1':
                return 1
            else:
                return -1

        return 0
    
    def GAME_OVER(self, state):
        result = OrbitoModel.GOAL_TEST(state)
        if len(result) > 0:
            return True
        return False

    def agent_function(self, observation, agent):
        action = self.MINIMAX(observation, agent)
        return action







# def MINIMAX(self, observation, agent):
#         self.player = agent
#         player = agent
#         best_value = -INFINITY if player else INFINITY
#         best_action = None

#         # Move Ordering
#         ordered_actions = self.move_ordering(observation, player)

#         for action, r in ordered_actions:
#             state_key = (tuple(map(tuple, observation['observation'])), tuple(action))
#             if state_key in self.state_action_dict:
#                 value = self.state_action_dict[state_key]
#             else:
#                 result = r
#                 next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
#                 value = self.MIN(next_state, 1, -INFINITY, INFINITY, self.players[0] if player != self.players[0] else self.players[1])
#                 self.state_action_dict[state_key] = value

#             if (player and value > best_value) or (not player and value < best_value):
#                 best_value = value
#                 best_action = action

#                 # Early Termination
#                 if (player and value == INFINITY) or (not player and value == -INFINITY):
#                     return best_action

#         return best_action

#     def MAX(self, current_state, depth, alpha, beta, player):
#         if depth >= self.cutoff_depth or self.GAME_OVER(current_state['observation']):
#             return self.EVALUATE(current_state)

#         best_value = -INFINITY

#         # Move Ordering
#         ordered_actions = self.move_ordering(current_state, player)

#         # Shallow Search
#         shallow_depth = 2
#         if depth < shallow_depth:
#             ordered_actions = ordered_actions[:len(ordered_actions)//2]

#         # Alpha-Beta Pruning
#         for action, r in ordered_actions:
#             state_key = (tuple(map(tuple, current_state['observation'])), tuple(action))
#             if state_key in self.state_action_dict:
#                 value = self.state_action_dict[state_key]
#             else:
#                 result = r
#                 next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
#                 value = self.MIN(next_state, depth + 1, alpha, beta, self.players[0] if player != self.players[0] else self.players[1])
#                 self.state_action_dict[state_key] = value

#             if value > best_value:
#                 best_value = value
#                 alpha = max(alpha, best_value)

#             if beta <= alpha:
#                 break  # Pruning

#         return best_value

#     def MIN(self, current_state, depth, alpha, beta, player):
#         if depth >= self.cutoff_depth or self.GAME_OVER(current_state['observation']):
#             return self.EVALUATE(current_state)

#         best_value = INFINITY

#         # Move Ordering
#         ordered_actions = self.move_ordering(current_state, player)

#         # Shallow Search
#         shallow_depth = 2
#         if depth < shallow_depth:
#             ordered_actions = ordered_actions[:len(ordered_actions)//2]

#         # Alpha-Beta Pruning
#         for action, r in ordered_actions:
#             state_key = (tuple(map(tuple, current_state['observation'])), tuple(action))
#             if state_key in self.state_action_dict:
#                 value = self.state_action_dict[state_key]
#             else:
#                 result = r
#                 next_state = OrbitoModel.ACTIONS(result, self.players[0] if player != self.players[0] else self.players[1])
#                 value = self.MAX(next_state, depth + 1, alpha, beta, self.players[0] if player != self.players[0] else self.players[1])
#                 self.state_action_dict[state_key] = value

#             if value < best_value:
#                 best_value = value
#                 beta = min(beta, best_value)

#             if beta <= alpha:
#                 break  # Pruning

#         return best_value

    
#     def move_ordering(self, state, player):
#         actions = []
#         manipulate_choices = [i for i, mask in enumerate(state['manipulate_mask']) if mask]
        
#         for i in manipulate_choices:
#             for j in state['manipulate_mask'][i]:
#                 possible_actions = np.setdiff1d(np.where(state['action_mask'])[0], j)
#                 possible_actions = np.append(possible_actions, i)
#                 for k in possible_actions:
#                     actions.append(((k, i, j, 1), OrbitoModel.RESULT(state['observation'], [k, i, j, 1], player)))
        
#         actions.extend([((i, 0, 0, 0), OrbitoModel.RESULT(state['observation'], [i, 0, 0, 0], player)) for i in np.where(state['action_mask'])[0]])
        
#         # Implement a heuristic to prioritize moves
#         # For example, prioritize moves that result in immediate wins or blocks
#         def heuristic(action):
#             result = action[1]
#             if self.GAME_OVER(result):
#                 return INFINITY if self.EVALUATE({'observation': result}) == 1 else -INFINITY
#             return 0
        
#         actions.sort(key=heuristic, reverse=True)
#         return actions