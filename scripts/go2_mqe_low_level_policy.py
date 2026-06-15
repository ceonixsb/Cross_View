# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import torch


class MQEGo2LowLevelPolicy:
    """MQE-style low-level locomotion policy (walk_these_ways adaptation+body)."""

    def __init__(self, num_envs: int, device: str, dt: float, locomotion_policy_dir: str):
        locomotion_dir = Path(locomotion_policy_dir).expanduser()
        body_path = locomotion_dir / "body_latest.jit"
        adaptation_path = locomotion_dir / "adaptation_module_latest.jit"
        if not body_path.exists():
            raise FileNotFoundError(f"MQE body policy not found: {body_path}")
        if not adaptation_path.exists():
            raise FileNotFoundError(f"MQE adaptation module not found: {adaptation_path}")

        self.device = torch.device(device)
        self.num_envs = num_envs
        self.dt = dt

        self.body = torch.jit.load(str(body_path), map_location=self.device)
        self.adaptation = torch.jit.load(str(adaptation_path), map_location=self.device)
        self.body.eval()
        self.adaptation.eval()

        self.locomotion_obs = torch.zeros((num_envs, 70), dtype=torch.float32, device=self.device)
        self.history_obs = torch.zeros((num_envs, 2100), dtype=torch.float32, device=self.device)
        self.last_action = torch.zeros((num_envs, 12), dtype=torch.float32, device=self.device)
        self.last_last_action = torch.zeros((num_envs, 12), dtype=torch.float32, device=self.device)
        self.clock_inputs = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.zeros((num_envs,), dtype=torch.float32, device=self.device)

        self._init_default_command_obs()

    def _init_default_command_obs(self):
        # Matches MQE Go2 default command slots when command.cfg.vel=True.
        # obs[3:5]=lin_vel slots, obs[5]=ang_vel slot (updated each step from high-level command).
        # fixed command terms from default command config:
        # gait_freq=3.0, gait=trotting([0.5, 0, 0], duration=0.5), footswing=0.08, stance_w=0.25, stance_l=0.45
        self.locomotion_obs[:, 6] = 0.0
        self.locomotion_obs[:, 7] = 3.0
        self.locomotion_obs[:, 8] = 0.5
        self.locomotion_obs[:, 9] = 0.0
        self.locomotion_obs[:, 10] = 0.0
        self.locomotion_obs[:, 11] = 0.5
        self.locomotion_obs[:, 12] = 0.08 * 0.15
        self.locomotion_obs[:, 13] = 0.0
        self.locomotion_obs[:, 14] = 0.0
        self.locomotion_obs[:, 15] = 0.25
        self.locomotion_obs[:, 16] = 0.45
        self.locomotion_obs[:, 17] = 0.0

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.history_obs[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self.last_last_action[env_ids] = 0.0
        self.clock_inputs[env_ids] = 0.0
        self.gait_indices[env_ids] = 0.0
        self.locomotion_obs[env_ids, 0:6] = 0.0
        self.locomotion_obs[env_ids, 18:70] = 0.0
        self._init_default_command_obs()

    def _step_clock_inputs(self):
        frequencies = self.locomotion_obs[:, 7]
        phases = self.locomotion_obs[:, 8]
        offsets = self.locomotion_obs[:, 9]
        bounds = self.locomotion_obs[:, 10]
        durations = self.locomotion_obs[:, 11]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]
        foot_indices = [torch.remainder(x, 1.0) for x in foot_indices]

        for idxs in foot_indices:
            stance = idxs < durations
            swing = idxs > durations
            idxs[stance] = idxs[stance] * (0.5 / torch.clamp(durations[stance], min=1e-6))
            idxs[swing] = 0.5 + (idxs[swing] - durations[swing]) * (
                0.5 / torch.clamp(1.0 - durations[swing], min=1e-6)
            )

        self.clock_inputs[:, 0] = torch.sin(2.0 * torch.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2.0 * torch.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2.0 * torch.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2.0 * torch.pi * foot_indices[3])

    @torch.no_grad()
    def step(
        self,
        projected_gravity: torch.Tensor,
        joint_pos_rel: torch.Tensor,
        joint_vel: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        # command slots with MQE scales: lin_vel*2.0, ang_vel*0.25
        self.locomotion_obs[:, 3:5] = command[:, 0:2] * 2.0
        self.locomotion_obs[:, 5] = command[:, 2] * 0.25

        self._step_clock_inputs()

        self.locomotion_obs[:, 0:3] = projected_gravity
        self.locomotion_obs[:, 18:30] = joint_pos_rel
        self.locomotion_obs[:, 30:42] = 0.05 * joint_vel
        self.locomotion_obs[:, 42:54] = self.last_action
        self.locomotion_obs[:, 54:66] = self.last_last_action
        self.locomotion_obs[:, 66:70] = self.clock_inputs

        self.history_obs = torch.cat((self.history_obs[:, 70:], self.locomotion_obs), dim=-1)
        latent = self.adaptation.forward(self.history_obs)
        action = self.body.forward(torch.cat((self.history_obs, latent), dim=-1))

        self.last_last_action = self.last_action.clone()
        self.last_action = action.clone()
        return action
