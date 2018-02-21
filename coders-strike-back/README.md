This project aims to experiment with some basic Reinforcement Learning algorithms (DQN for now) using TensorFlow.
The test environment chosen is a mini-game called 'Coders Strike Back', from the website CodinGame. I saw potential in this game to apply
some RL algorithms since it has a well defined environment simulating some basic physics and presents a rich set of actions for the agent.

Basically, the agent must pass sequentially through all the checkpoints generated in the map 3 times to win the race. It can observe its
own location, orientation and velocity as well as the location of all checkpoints, and it can perform turns of maximum 18 degrees
at each game step and can accelerate with different values.

The game can also include other agents, such as an opponent in a 1vs1 race or a teammate and two opponents in a 2vs2 race in which several 
strategies can be developed, such as using one agent to block the opponents and the other to run through the checkpoints dodging the opponents.
This is quite more advanced though, and for now this project is limited to just 1 agent trying to finish the race as fast as possible.

A sample of one of the best results so far is shown in the 'ani' folder (standing for animation). It turned out to be more difficult to train than I expected,
so it can clearly be seen that a lot of progress is still needed to achieve a decent runner agent.
