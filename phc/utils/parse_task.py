# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from phc.env.tasks.humanoid import Humanoid
from phc.env.tasks.humanoid_amp import HumanoidAMP
from phc.env.tasks.humanoid_amp_getup import HumanoidAMPGetup
from phc.env.tasks.humanoid_im import HumanoidIm
from phc.env.tasks.humanoid_im_getup import HumanoidImGetup
from phc.env.tasks.humanoid_im_mcp import HumanoidImMCP
from phc.env.tasks.humanoid_im_mcp_getup import HumanoidImMCPGetup
from phc.env.tasks.vec_task_wrappers import VecTaskPythonWrapper
from phc.env.tasks.humanoid_im_demo import HumanoidImDemo
from phc.env.tasks.humanoid_im_mcp_demo import HumanoidImMCPDemo

from isaacgym import rlgpu

import json
import numpy as np

from rofunc.config.utils import get_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers
from rofunc.learning.utils.utils import set_seed
from rofunc.learning.utils.env_wrappers import wrap_env
def warn_task_name():
    raise Exception("Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")



def parse_task(args, cfg, cfg_train, sim_params):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    # task = eval(args.task)(cfg=cfg, sim_params=sim_params, physics_engine=args.physics_engine, device_type=args.device, device_id=device_id, headless=args.headless)
    # env = VecTaskPythonWrapper(task, rl_device, cfg_train.get("clip_observations", np.inf))
    task = None

    custom_args = {
        "task": "env_im_g1_phc",
        "train": "HumanoidPHCLargeRofuncRL",
        "use_pnn": True,
        "humanoid_robot_type": "unitree_g1",
        "num_envs": 1024,
        "sim_device": 0,
        "rl_device": 0,
        "control": "robot_control",
        "sim": "robot_sim",
        "fixed_log_std": -1.7,
        "headless": True,
        "inference": False,
        "debug": False,
    }
    from omegaconf import DictConfig, OmegaConf
    from easydict import EasyDict

    custom_args = DictConfig(custom_args)

    args_overrides = [f"task={custom_args.task}",
                      f"train={custom_args.train}",
                      f"robot={custom_args.humanoid_robot_type}",
                      f"control={custom_args.control}",
                      f"sim={custom_args.sim}",
                      f"device_id={custom_args.sim_device}",
                      f"rl_device=cuda:{custom_args.rl_device}",
                      f"headless={custom_args.headless}",
                      f"num_envs={custom_args.num_envs}"]

    # env_name = f"{custom_args.task}_{custom_args.humanoid_robot_type}"
    cfg, rofunc_logger = get_config("./learning/rl", "config", args=args_overrides, )
    cfg.task.env.numEnvs = custom_args.num_envs
    cfg.train.Model.fixed_log_std = custom_args.fixed_log_std
    if custom_args.debug == "True":
        cfg.train.Trainer.wandb = False
    cfg.train.Model.use_pnn = custom_args.use_pnn
    cfg_dict = EasyDict(OmegaConf.to_container(cfg, resolve=True))
    set_seed(cfg.train.Trainer.seed)

    # Instantiate the Isaac Gym environment
    env = Tasks().task_map[custom_args.task](cfg=cfg.task,
                                             rl_device=cfg.rl_device,
                                             sim_device=f'cuda:{cfg.device_id}',
                                             graphics_device_id=cfg.device_id,
                                             headless=cfg.headless,
                                             virtual_screen_capture=cfg.capture_video,  # TODO: check
                                             force_render=cfg.force_render,
                                             rofunc_logger=rofunc_logger)

    # env = wrap_env(env, logger=rofunc_logger, seed=cfg.train.Trainer.seed)

    return task, env
