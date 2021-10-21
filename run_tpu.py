from absl import flags
import argparse
from pysc2.env import sc2_env
import sys
import torch
from Utils import train, utils
from RelationalModule import CoordActorCritic
import torch_xla.core.xla_model as xm

parser = argparse.ArgumentParser(description='A2C for StarCraftII minigames')
# Game arguments
parser.add_argument('--res', type=int, help='Screen and minimap resolution', default=16)

args, unknown_flags = parser.parse_known_args()  # Let argparse parse known flags from sys.argv.
flags.FLAGS(sys.argv[:1] + unknown_flags)  # Let absl.flags parse the rest.

def init_agent(tpu=True):

    HPs = dict(action_space=3, observation_space=6, lr=0.0005, gamma=0.9, TD=True, 
                 twin=True, tau=0.1, n_steps=40, H=1e-3, hiddens=[64,32,16])

    if tpu:
        dev = xm.xla_device()
        HPs['device'] = dev
    else:
        if torch.cuda.is_available():
            HPs['device'] = 'cuda'
        else:
            HPs['device'] = 'cpu'
            
        print("Using device "+HPs['device'])
        
    agent = CoordActorCritic.MoveToBeaconA2C(**HPs)

    return agent

def main():
    # Environment parameters
    RESOLUTION = args.res
    game_params = dict(feature_screen=RESOLUTION, feature_minimap=RESOLUTION, action_space="FEATURES") 

    replay_dict = dict(save_replay_episodes=1000,
                   replay_dir='Replays/',
                   replay_prefix='Agent1')

    MAX_STEPS = 256
    N_EPISODES = 1000

    agent = init_agent()

    results = train.train_SC2_agent(agent, game_params, N_EPISODES, MAX_STEPS, return_agent=True,  **replay_dict)
    
    return

if __name__=="__main__":
    main()