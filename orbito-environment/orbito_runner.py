#!/usr/bin/env python3

import gymnasium as gym
import orbito.orbito.orbito_v0 as orbito_v0
import argparse
import logging
import sys
import time

def create_environment(render_mode, seed=None):
    env = orbito_v0.env(render_mode=render_mode)
    if seed:
        env.reset(seed)
    return env

def destroy_environment(env):
    env.close()
    return

def run_one_episode(env, agent1, agent2):
    agents = { 'player_0': agent1, 'player_1': agent2 }
    times = { "player_0": 0.0, "player_1": 0.0 }
    env.reset()
    agent1.reset()
    agent2.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
            break

        t1 = time.time()
        action = agents[agent].agent_function(observation, agent)
        t2 = time.time()
        times[agent] += (t2-t1)
        env.step(action)

    winner = None
    if len(env.rewards.keys()) == 2:
        for a in env.rewards:
            if env.rewards[a] == 1:
                winner = a
                break

    # for agent in times:
    #     print(f"{agent} took {times[agent]:8.5f} seconds.")

    return winner

def run_many_episodes(env, episode_count, agent1, agent2):
    agents = ['player_0', 'player_1']
    winners = {}
    for i in range(episode_count):
        winner = run_one_episode(env, agent1, agent2)
        print(f"Episode {i+1}: {winner}")
        if winner is None:
            winner = "draw"
        if winner not in winners:
            if winner is None:
                winner = "draw"
            winners[winner] = 0
        winners[winner] += 1
    destroy_environment(env)
    return winners

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Run Connect Four')
    parser.add_argument(
        "--episode-count",
        "-c",
        type=int, 
        help="number of episodes to run",
        default=1
    )
    parser.add_argument(
        "--logging-level",
        "-l",
        type=str,
        help="logging level: warn, info, debug",
        choices=("warn", "info", "debug"),
        default="warn",
    )
    parser.add_argument(
        "--seed",
        type=int, 
        help="seed for the environment's PRNG",
        default=0
    )
    parser.add_argument(
        "--render-mode",
        "-r",
        type=str,
        help="display style (render mode): human, none",
        choices=("human", "none"),
        default="human",
    )
    parser.add_argument(
        "--agent1",
        "-a",
        type=str,
        help="agent function: random, agent1, human, agent, agent2",
        choices=("random", "agent1", "human", "agent", "agent2"),
        default="random",
    )
    parser.add_argument(
        "--agent2",
        "-A",
        type=str,
        help="agent function: random, agent1, human, agent, agent2",
        choices=("random", "agent1", "human", "agent", "agent2"),
        default="random",
    )

    my_args = parser.parse_args(argv[1:])
    if my_args.logging_level == "warn":
        my_args.logging_level = logging.WARN
    elif my_args.logging_level == "info":
        my_args.logging_level = logging.INFO
    elif my_args.logging_level == "debug":
        my_args.logging_level = logging.DEBUG

    if my_args.render_mode == "none":
        my_args.render_mode = None
    return my_args

from orbito_demo_agents.agent_random import AgentRandom
from orbito_demo_agents.agent1 import OrbitoAgent, OrbitoAgentV1, OrbitoAgentV2
from orbito_demo_agents.human import AgentHuman

def select_agent(agent_name):
    if agent_name == "random":
        agent_function = AgentRandom()
    elif agent_name == "agent":
        agent_function = OrbitoAgent()
    elif agent_name == "human": 
        agent_function = AgentHuman()
    elif agent_name == "agent1":
        agent_function = OrbitoAgentV1()
    elif agent_name == "agent2":
        agent_function = OrbitoAgentV2()
    else:
        raise Exception(f"unknown agent name: {agent_name}")
    return agent_function

def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=my_args.logging_level)

    env = create_environment(my_args.render_mode, my_args.seed)
    agent1 = select_agent(my_args.agent1)
    agent2 = select_agent(my_args.agent2)
    winners = run_many_episodes(env, my_args.episode_count, agent1, agent2)
    print(f"Winners: {winners}")
    return

if __name__ == "__main__":
    main(sys.argv)
