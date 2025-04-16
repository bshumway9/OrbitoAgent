#!/usr/bin/env python3

import random
import numpy as np

class AgentRandom:

    def __init__(self):
        """required method"""
        return

    def reset(self):
        """required method"""
        return

    def agent_function(self, observation, agent):
        """required method"""
        #finds the possible opponent pieces to manipulate
        manipulate_choices = []
        for i in range(len(observation['manipulate_mask'])):
            if observation['manipulate_mask'][i]:
                manipulate_choices.append(i)
        random_index = random.choice(manipulate_choices) if len(manipulate_choices) > 0 else None
        #randomly selects where to move the opponents piece out of available adjacent spaces
        manipulate_action = random.choice(observation['manipulate_mask'][random_index]) if random_index is not None else None
        manipulate = 0
        if random_index is not None:
            manipulate = 1
        #randomly selects a move out of available moves which has been adjusted for the manipulation
        possible_actions = np.where(observation['action_mask'])[0]
        if manipulate:
            possible_actions = np.delete(possible_actions, np.where(possible_actions == manipulate_action))
            possible_actions = np.append(possible_actions, random_index)
        action = np.random.choice(possible_actions)
        
        return [int(action), random_index if random_index is not None else 0, manipulate_action if manipulate_action is not None else 0, manipulate]