# rlcirclebot_simulator
The purpose of this project is to practice RL and Gymnasium and SB3 APIs.
The goal of this project was to create a simulator where a virtual circle-shaped bot with skid-steer/tank drive and collision physics could navigate a virtual space with 
obstacles. For simplicity, the space is a square grid with obstacles as 1m x 1m unit squares. Then an RL model would be used to predict the motor powers to navigate the robot
from the start point to the endpoint.

The RL model has access to the bot's current forward and lateral speed(in the robot's POV), the distance and angle from the target, as well as 11 ray-projections equally spaced
from (-PI/2, PI/2) to act as a low-res lidar sensor. Since this is a skid steer bot, it would predict the power of left and right motors. The model also learns in 3 stages, the first
stage has obstacles near the path, but not in the way of getting to the target. The second has low-frequency obstacles in the way of the robot, and the last stage has obstacles
in high frequencies.

This repo includes the following classes
grid.py - stores the map, obstacles, collision detection, and logic for ray projection
circlebot.py - stores the current state of the robot, simulation of 1 time step, skid-steer physics, and collision physics
circlebot_env.py - the Gymnasium environment for the bot, the logic for generation of the maps for the 3 stages, logic to render one frame, and reward calculation
circlebot_sim.py - tests the env, trains the model, selects the stage, saves the weights and vecnorm, and renders a mp4 of one result
robot_simX.mp4 - An evaluation after the training of stage X
ppo_stageX.zip - the weights after the training of stage X
circlebot_vecnormX.pkl - the vector normalization after training of stage X
