# run_actor.py

import sys
import os

from envs.lunarlander_v3.env import Env
# from envs.cartpole_v1.env import Env
# 导入 Actor 类
from actor.actor_stream import Actor

if __name__ == "__main__":
    actor = Actor(Env(4),"localhost",50051)
    actor.run()