import gymnasium as gym
from gymnasium.utils.play import play

play(gym.make('ALE/Galaxian-v5', render_mode='rgb_array'),
     keys_to_action={'w': 1, 'd': 2, 'a': 3, '.': 4, ',': 5}, noop = 0)
     
#1 = Fire
#2 = Move Right
#3 = Move Left
#4 = Move Right and Fire
#5 = Move Left and Fire
