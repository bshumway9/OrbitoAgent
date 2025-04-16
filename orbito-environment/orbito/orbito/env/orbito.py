# #!/usr/bin/env python3

from __future__ import annotations

import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import agent_selector as AgentSelector

from orbito.env.orbito_model import OrbitoModel, rotatePieces, to1by16, to4by4

def get_image(path):
    from os import path as os_path

    import pygame

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def env(**kwargs):
    env = raw_env(**kwargs)
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "orbito_v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode: str | None = None, screen_scaling: int = 9):
        EzPickle.__init__(self, render_mode, screen_scaling)
        super().__init__()
        # 6 rows x 7 columns
        # blank space = 0
        # agent 0 -- 1
        # agent 1 -- 2
        # flat representation in row major order
        self.screen = None
        self.render_mode = render_mode
        self.screen_scaling = screen_scaling

        self.board = [[0] * 4 for _ in range(4)]

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: MultiDiscrete([16, 16, 16, 2]) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(low=0, high=2, shape=(4,4), dtype=np.int8),
                    "action_mask": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8),
                }
            )
            for i in self.agents
        }
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    # Key
    # ----
    # blank space = 0
    # agent 0 = 1
    # agent 1 = 2
    # An observation is list of lists, where each list represents a row
    #
    # array([[0, 1, 1, 2],
    #        [1, 0, 1, 2],
    #        [0, 1, 0, 0],
    #        [1, 1, 2, 1]], dtype=int8)
    def observe(self, agent):
        return OrbitoModel.ACTIONS(self.board, agent)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def encode_values(self, val1, val2, val3):
        total = val1
        total = total * 17 + val2
        total = total * 17 + val3
        return total

    def decode_values(self, encoded_val):
        val3 = encoded_val % 17
        encoded_val = encoded_val // 17
        val2 = encoded_val % 17
        encoded_val = encoded_val // 17
        val1 = encoded_val
        return val1, val2, val3

    def _legal_moves(self):
        return [i for i in range(16) if self.board[i] == 0]
    
    # the available actions for the current player to move the opponents piece horizontally or vertically by one space
    def _legal_manipulate_pieces(self, opp_p_board):
        positions = [i for i in range(16) if opp_p_board.flatten()[i]]
        legal_moves = [[] for x in range(16)]
        for i in positions:
            if i % 4 != 0 and self.board[i-1] == 0:
                legal_moves[i].append(i-1)
            if i % 4 != 3 and self.board[i+1] == 0:
                legal_moves[i].append(i+1)
            if i > 3 and self.board[i-4] == 0:
                legal_moves[i].append(i-4)
            if i < 12 and self.board[i+4] == 0:
                legal_moves[i].append(i+4)
        return legal_moves

    # action in this case is a value from 0 to 15 indicating position to move on the orbito board
    def step(self, action_data):
        # action, manipulate_piece, manipulate_action = self.decode_values(action_data)
        action, manipulate_piece, manipulate_action, manipulate = (action_data[0], action_data[1], action_data[2], action_data[3])
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        if self.render_mode == "human":
            self.board = OrbitoModel.human_RESULT_1(self.board, action_data, self.agent_selection)
            self.render()
            self.board = OrbitoModel.human_RESULT_2(self.board, action_data, self.agent_selection)
            self.render()
            self.board = OrbitoModel.human_RESULT_3(self.board)
            self.render()
        else:
            self.board = OrbitoModel.RESULT(self.board, action_data, self.agent_selection)
        

        winner = OrbitoModel.GOAL_TEST(self.board)
    
        # check if there is a winner
        if winner:
            #check for tie
            if len(winner) == 2:
                self.rewards['player_0'] = 0
                self.rewards['player_1'] = 0
            elif winner[0] == 0:
                self.rewards['player_0'] += 1
                self.rewards['player_1'] -= 1
            elif winner[0] == 1:
                self.rewards['player_0'] -= 1
                self.rewards['player_1'] += 1
            self.terminations = {i: True for i in self.agents}
        # check if board is full, if so rotate the board 5 times to check for a winner before tie is declared
        elif all(item in [1, 2] for sublist in self.board for item in sublist):
            for i in range(5):
                board = to1by16(self.board.copy())
                board = rotatePieces(board)
                self.board = to4by4(board)
                if self.render_mode == "human":
                    self.render()
                winner = OrbitoModel.GOAL_TEST(self.board)
                if winner:
                    #check for tie
                    if len(winner) == 2:
                        self.rewards['player_0'] = 0
                        self.rewards['player_1'] = 0
                    elif winner[0] == 0:
                        self.rewards['player_0'] += 1
                        self.rewards['player_1'] -= 1
                    elif winner[0] == 1:
                        self.rewards['player_0'] -= 1
                        self.rewards['player_1'] += 1
                    self.terminations = {i: True for i in self.agents}
                    break
            # once either play wins or there is a draw, game over, both players are done
            self.terminations = {i: True for i in self.agents}
        next_agent = self._agent_selector.next()
        self.agent_selection = next_agent

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def rotatePieces(self):
        new_board = self.board.copy()
        for i in [1,2,3]:
            new_board[i-1] = self.board[i]
        for i in [0,4,8]:
            new_board[i+4] = self.board[i]
        for i in [12,13,14]:
            new_board[i+1] = self.board[i]
        for i in [15,11,7]:
            new_board[i-4] = self.board[i]
        new_board[5] = self.board[6]
        new_board[9] = self.board[5]
        new_board[10] = self.board[9]
        new_board[6] = self.board[10]
        self.board = new_board

    def reset(self, seed=None, options=None):
        # reset environment
        self.board = [[0] * 4 for _ in range(4)]

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._agent_selector = AgentSelector(self.agents)

        self.agent_selection = self._agent_selector.reset()

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        screen_width = 99 * self.screen_scaling
        screen_height = screen_width

        if self.screen is None:
            pygame.init()
        
            if self.render_mode == "human":
                pygame.display.set_caption("Orbito")
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((screen_width, screen_height))

        # Load and scale all of the necessary images
        tile_size = (screen_width * (91 / 99)) / 4

        white_chip = get_image(os.path.join("img", "C4WhitePiece.png"))
        white_chip = pygame.transform.scale(
            white_chip, (int(tile_size * (9 / 13)), int(tile_size * (9 / 13)))
        )

        black_chip = get_image(os.path.join("img", "C4BlackPiece.png"))
        black_chip = pygame.transform.scale(
            black_chip, (int(tile_size * (9 / 13)), int(tile_size * (9 / 13)))
        )

        board_img = get_image(os.path.join("img", "OrbitoBoard.png"))
        board_img = pygame.transform.scale(
            board_img, ((int(screen_width)), int(screen_height))
        )

        self.screen.blit(board_img, (0, 0))

        # Blit the necessary chips and their positions
        flattened_board = [item for sublist in self.board for item in sublist]
        for i in range(0, 16):
            if flattened_board[i] == 1:
                self.screen.blit(
                    white_chip,
                    (
                        (i % 4) * (tile_size) + (tile_size * (6 / 13)),
                        int(i / 4) * (tile_size) + (tile_size * (6 / 13)),
                    ),
                )
            elif flattened_board[i] == 2:
                self.screen.blit(
                    black_chip,
                    (
                        (i % 4) * (tile_size) + (tile_size * (6 / 13)),
                        int(i / 4) * (tile_size) + (tile_size * (6 / 13)),
                    ),
                )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def check_for_winner(self):
        board = np.array(self.board).reshape(4, 4)
        piece = self.agents.index(self.agent_selection) + 1

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
                    return True

        # Check vertical locations for win
        for c in range(column_count):
            for r in range(row_count - 3):
                if (
                    board[r][c] == piece
                    and board[r + 1][c] == piece
                    and board[r + 2][c] == piece
                    and board[r + 3][c] == piece
                ):
                    return True

        # Check positively sloped diagonals
        for c in range(column_count - 3):
            for r in range(row_count - 3):
                if (
                    board[r][c] == piece
                    and board[r + 1][c + 1] == piece
                    and board[r + 2][c + 2] == piece
                    and board[r + 3][c + 3] == piece
                ):
                    return True

        # Check negatively sloped diagonals
        for c in range(column_count - 3):
            for r in range(3, row_count):
                if (
                    board[r][c] == piece
                    and board[r - 1][c + 1] == piece
                    and board[r - 2][c + 2] == piece
                    and board[r - 3][c + 3] == piece
                ):
                    return True

        return False
    

if __name__ == "__main__":
    env = raw_env()
    env.reset()
    env.render()