#!/usr/bin/env python3

import random
import numpy as np

class AgentHuman:

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

        manipulate = 0
        if len(manipulate_choices) > 0:
            while True:
                try:
                    index = int(input(f"Choose an opponents piece from the available pieces {manipulate_choices}: "))
                    if index in manipulate_choices:
                        manipulate = 1
                        break
                    else:
                        print(f"Invalid piece. Please choose a valid piece {manipulate_choices}: ")
                except ValueError:
                    print(f"Invalid input. Please enter a number {manipulate_choices}: ")
        
        if manipulate:
            while True:
                try:
                    manipulate_action = int(input(f"Choose a place to move opponents piece from the available options {observation['manipulate_mask'][index]}: "))
                    if manipulate_action in observation['manipulate_mask'][index]:
                        break
                    else:
                        print(f"Invalid action. Please choose a valid option {observation['manipulate_mask'][index]}: ")
                except ValueError:
                    print(f"Invalid input. Please enter a number {observation['manipulate_mask'][index]}: ")


        possible_actions = np.where(observation['action_mask'])[0]
        if manipulate:
            possible_actions = np.delete(possible_actions, np.where(possible_actions == manipulate_action))
            possible_actions = np.append(possible_actions, index)

        while True:
            try:
                action = int(input(f"Choose an action from the available options {possible_actions}: "))
                if action in possible_actions:
                    break
                else:
                    print(f"Invalid action. Please choose a valid action {possible_actions}: ")
            except ValueError:
                print(f"Invalid input. Please enter a number {possible_actions}: ")
        if manipulate:
            return [int(action), index, manipulate_action, manipulate]
        else:
            return [int(action), 0, 0, manipulate]