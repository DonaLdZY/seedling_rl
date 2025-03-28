
from envs.cartpole_v1.env import Env
from actor.actor_stream import Actor

if __name__ == "__main__":
    actor = Actor(Env(1),"localhost",50051)
    actor.run()