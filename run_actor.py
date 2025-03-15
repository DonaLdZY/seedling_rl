# run_actor.py

import sys
import os

# 将项目根目录添加到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入 Actor 类
from actor.CartPole_v1.actor import Actor

if __name__ == "__main__":
    actor = Actor("localhost",50051)
    actor.start()