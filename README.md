# Galaxian Reinforcement Learning
This project is a reinforcement learning based AI that plays the Atari game Galaxian. Below one can see a video of the Advantage Actor Critic (A2C) agent playing the game after being trained



<img height="500" img width='400' src=https://user-images.githubusercontent.com/94200328/234165993-67c387cf-3f88-4c74-a133-210a1cc7008b.gif>

## Motivation
The goal of this project is to explore the use of reinforcement learning to play classic Atari games such as Galaxian. My aim was to create agents that can learn to play these games at a high level without being explicitly programmed to do so. Reinforcement learning has the potential to enable machines to learn from experience and make decisions in complex environments. By applying reinforcement learning to games, one can develop agents that are capable of learning to solve complex problems and provide insights into the effectiveness of different algorithms and approaches.

Through this project, I hope to demonstrate the capabilities and limitations of different reinforcement learning algorithms such as A2C and DQN, and provide a framework for future research and development in this field. This project can serve as a starting point for anyone interested in learning about reinforcement learning, deep learning, or game development, and provides a practical example of how these concepts can be applied to real-world problems.

## Experiments
In this project, I investigated the effectiveness of two reinforcement learning algorithms, A2C and DQN, for playing the classic arcade game Galaxian. Our goal was to evaluate the agents' ability to learn to play the game at a high level without being explicitly programmed to do so, and to compare the performance of the two algorithms under different settings.

### Advantage Actor Critic (A2C)
For the A2C agent, I focused on measuring its average episode return, actor and critic losses, and entropy, as well as its ability to avoid collisions with the aliens.

![a2c_nenvs500_neps2000_spu128_clr0 00001_alr0 0005](https://user-images.githubusercontent.com/94200328/234167564-7585fc73-c6e6-41eb-a22a-2bdd307d7efa.png)

### Deep Q-Network (DQN)
For the DQN agent, I focused on measuring its average episode reward and episode duration, as well as its ability to complete the game quickly.

![Duration_vs_epis_lr0 0001_tau0 0005](https://user-images.githubusercontent.com/94200328/234167622-563ff5e6-f038-4055-a53a-cf74f9982fa3.png)
![rewards_vs_epis_lr0 0001_tau0 0005](https://user-images.githubusercontent.com/94200328/234167650-4e7e4399-b7ae-439e-942b-f72117b3f9d4.png)

## Challenges
Some challenges I had while doing this project were that with the little knowledge I had about reinforcement learning coming into this it was difficult to find code for a DQN agent that I could change to play Galaxian on. I kept running into errors with layer sizes and things of that nature that I was able to fix for the A2C agent, but for some reason proved more difficult to fix in the DQN agent. This resulted in me not being able to get as much in terms of visualization from the DQN agent. I would like to make it so in the future I could record a video of the trained DQN agent playing the game and also be able to plot the loss over time from the DQN agent, but could not figure it out in the time I had to do the project so far.

## Conclusions
Through our experiments, I found that both A2C and DQN were able to learn effective policies for playing Galaxian, but had different strengths and weaknesses. The A2C agent was able to steadily increase its average episode return over the course of 2000 updates, but struggled to avoid collisions with the aliens. The DQN agent, on the other hand, was able to achieve a consistent level of success in the game, but may have had difficulty improving its performance over time.

Overall, our experiments highlight the potential of reinforcement learning algorithms for playing Atari games like Galaxian, and suggest that further research is needed to better understand the strengths and limitations of different algorithms under different settings.

## Think you can do better than the model??
<img width="650" img height='600' alt="Screen Shot 2023-04-25 at 11 58 44 PM" src="https://user-images.githubusercontent.com/94200328/234466446-f166490c-bce4-4e89-97d8-3633b2b35af0.png">


Use the play.py file to play galaxian yourself and see if you can do better than the model can

## Important Notes
To use that Atari games you will need to install ALE-py from gymnasium as well as the AutoROM and the roms to play the games with. To do this you can use the code below
```
pip install 'gymnasium[atari]'
```
```
pip install 'gymnasium[accept-rom-license]'
```
Please note that unless you specify a location for the accept-rom-license command the roms will download in the directory you are currently in. To install the roms in a specific location you can go to this link for more details [AutoROM](https://github.com/Farama-Foundation/AutoROM).

Also note that I needed to have the quotes around gymnasium[atari] and gymnasium[accept-rom-license] for this to work in terminal on a macbook, but the quotes may not be needed if not using terminal.
