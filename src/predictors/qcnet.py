# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from modules import QCNetDecoder
from modules import QCNetEncoder
from utils import wrap_angle

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class QCNet(pl.LightningModule):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 submission_dir: str,
                 submission_file_name: str,
                 **kwargs) -> None:
        super(QCNet, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name

        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.decoder = QCNetDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')

        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)

        self.test_predictions = dict()
        self.pos_std: float = 0.0
        self.heading_std: float = 0.0

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred

    def training_step(self,
                      data,
                      batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        loss = reg_loss_propose + reg_loss_refine + cls_loss
        return loss

    def validation_step(self,
                        data,
                        batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        
        #? Add random noise to future position, heading, and velocity of agents
        for key in ('position', 'heading', 'velocity'):
            shape = list(data['agent'][key].shape)
            shape[1] = self.num_historical_steps
            if key == 'heading':
                noise_std = self.heading_std
            else:
                noise_std = self.pos_std

            noise = torch.normal(mean=torch.zeros(shape), std=noise_std).to(self.device)
            data['agent'][key][:, :self.num_historical_steps, ...] += noise

        #? Wrap the angle to [-pi, pi] using wrap_angle
        data['agent']['heading'][:, :self.num_historical_steps] = \
            wrap_angle(data['agent']['heading'][:, :self.num_historical_steps])
        
        pred = self(data)
        if self.output_head:    # False
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]



        def check_out_of_map(scenario_id, data, target_agent_traj, car_heading_on_map, car_pos_on_map, visualize=False):
            '''
                target_agent_traj: shape [60, 2]
            '''

            from av2.map.map_api import ArgoverseStaticMap
            from pathlib import Path
            import matplotlib.pyplot as plt
            import numpy as np
            static_map_path = (Path(f"/home/letian/UofTCourse/Motion_Planning/project/val/raw/{scenario_id}") / f"log_map_archive_{scenario_id}.json")
            static_map = ArgoverseStaticMap.from_json(static_map_path)

            # create polygon        
            paths = []
            from matplotlib.path import Path
            for drivable_area in static_map.vector_drivable_areas.values():
                for polygon in [drivable_area.xyz]:
                    paths.append(Path(polygon[:,:2]))

            # transform the target_agent trajectory to the map coodinates
            # cur_heading = data['agent']['heading'][eval_mask][:,50:51].item()
            # cur_pos = data['agent']['position'][eval_mask][0,50:51,:2].cpu().numpy()
            # import pdb; pdb.set_trace()
            transform_matrix = np.array([
                [np.cos(car_heading_on_map), -np.sin(car_heading_on_map)],
                [np.sin(car_heading_on_map), np.cos(car_heading_on_map)]
            ])
            target_agent_traj_in_map = (transform_matrix @ target_agent_traj.T).T + car_pos_on_map

            # check if the traj is out of drivable area
            out_of_drivable_area = False
            for one_point in target_agent_traj_in_map:
                if not any(path.contains_point(one_point) for path in paths):
                    out_of_drivable_area = True

            # visualize
            if visualize:
                for drivable_area in static_map.vector_drivable_areas.values():
                    for polygon in [drivable_area.xyz]:
                        plt.fill(polygon[:, 0], polygon[:, 1], color='0')
                plt.plot(target_agent_traj_in_map[:,0], target_agent_traj_in_map[:,1])
                plt.show()
                plt.close()
                # import pdb; pdb.set_trace()

            return out_of_drivable_area
        def check_small_turning_radius(trajectory, radius_threhold=3.5, minimum_radius=True):
            import numpy as np
            def calculate_curvature(x, y):
                dx_dt = np.gradient(x)
                dy_dt = np.gradient(y)
                d2x_dt2 = np.gradient(dx_dt)
                d2y_dt2 = np.gradient(dy_dt)
                curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / ((dx_dt ** 2 + dy_dt ** 2) ** 1.5)
                return curvature
            def calculate_turning_radius(curvature):
                return 1.0 / curvature

            # Example trajectory (replace with your actual trajectory)
            # x = np.array([0, 1, 2, 3, 4])
            # y = np.array([0, 1, 0, -1, 0])
            x = trajectory[:, 0]
            y = trajectory[:, 1]

            # Calculate curvature
            curvature = calculate_curvature(x, y)

            # Calculate turning radius
            turning_radius = calculate_turning_radius(curvature)

            # Print turning radius at each point along the trajectory
            # for i, radius in enumerate(turning_radius):
            #     print(f"Point {i + 1}: Turning radius = {radius}")

            # You can also calculate the average turning radius if needed
            average_radius = np.mean(np.abs(turning_radius))
            min_radius = np.min(np.abs(turning_radius))

            # print(f"Average turning radius = {average_radius}")
            # print(f"Minimum turning radius = {min_radius}")

            # import pdb; pdb.set_trace()
            if minimum_radius:
                if min_radius < radius_threhold: 
                    return True
                else:
                    return False
            else:
                if average_radius < radius_threhold: 
                    return True
                else:
                    return False

        # ''' check the most-likely trajectory '''
        # scenario_id = data['scenario_id'][0]
        # eval_mask = data['agent']['category'] == 3
        # target_agent_traj = traj_refine_best[eval_mask][0, :, :2].cpu().numpy()
        # # out_of_drivable_area = check_out_of_map(scenario_id, data, target_agent_traj, visualize=False)
        # # small_turning_radius = check_small_turning_radius(target_agent_traj)


        # ''' generate a trajectory via a planner, check if it's out of map '''
        # # ''' example to use the parameterized motion planning model '''
        # ood_switch_to_

        from planning_model import motion_skill_model
        import numpy as np
        num_car = data['agent']['position'][eval_mask].shape[0]
        num_modality = traj_eval.shape[1]
        traj_eval_from_planner = torch.zeros_like(traj_eval)
        pi_eval_from_planner = torch.ones_like(pi_eval)
        traj_eval_ood_switch = torch.zeros_like(traj_eval)
        pi_eval_ood_switch = torch.zeros_like(pi_eval)

        ''' ood classifier: detect out-of-map and over-small-radius '''
        ood_flag = torch.zeros(num_car)
        for i in range(num_car):
            # import pdb; pdb.set_trace()
            scenario_id = data['scenario_id'][i]
            eval_mask = data['agent']['category'] == 3

            ood_count_for_one_car = 0
            for modal_id in range(num_modality):
                traj = traj_eval[i, modal_id][:, :2].cpu().numpy()
                cur_heading_on_map = data['agent']['heading'][eval_mask][i,50:51].item()
                cur_pos_on_map = data['agent']['position'][eval_mask][i,50:51,:2].cpu().numpy()
                out_of_map = check_out_of_map(scenario_id, data, traj, cur_heading_on_map, cur_pos_on_map, visualize=False)
                small_turning_radius = check_small_turning_radius(traj)
                if out_of_map and small_turning_radius: ood_count_for_one_car += 1
            
            # half of the prediction is wrong
            if ood_count_for_one_car >= modal_id / 2: ood_flag[i] = 1

        ''' planning model '''
        for i in range(num_car):
            # current_v = 0
            current_v = np.sqrt(np.sum(np.square(data['agent']['position'][eval_mask][i,50:51,:2].cpu().numpy() - data['agent']['position'][eval_mask][i,49:50,:2].cpu().numpy()))) * 10
            # import pdb; pdb.set_trace()
            current_a = 0
            eval_mask = data['agent']['category'] == 3

            horizon = 60
            # planning_param_set = [[current_v * horizon / 10, 0, 0, current_v, 0],       # constant speed
            #                      [4 * horizon / 10, 0, 0, 4, 0],                # low speed
            #                      [8 * horizon / 10, 0, 0, 8, 0],                # medium speed
            #                      [15 * horizon / 10, 0, 0, 12, 0],                # high speed
            #                      [6 * horizon / 10, 6 * horizon / 10 / 2, 30, 6, 0],    # turn left
            #                      [6 * horizon / 10, -6 * horizon / 10 / 2, -30, 6, 0],  # turn right
            # ]

            # k-means clustering from data
            planning_param_set = [[ 2.3058292e+01,  2.3649578e+01,  6.9302147e+01,  3.8323116e+00, -9.1827745e+00],
                                  [ 7.6472034e+00, -8.9843893e-01, -3.4655058e+00,  1.1500170e+00, -2.4785318e+00],
                                  [ 4.6849834e+01, -2.2653617e-01, -2.4802389e-02,  3.8668215e+00, -9.4297333e+00],
                                  [ 6.7266579e+01, -1.4482324e-01,  1.4982077e-01,  5.5946622e+00, -1.3733204e+01],
                                  [ 2.6835514e+01, -3.7002566e+00, -1.0276335e+01,  2.4907744e+00, -5.7795033e+00],
                                  [ 9.5007362e+01, -5.2219313e-01, -5.8250517e-02,  7.9314003e+00, -1.9205318e+01]]



            for modal_id, (lon1, lat1, yaw1, v1, acc1) in enumerate(planning_param_set):
                print(lon1, lat1, yaw1, v1, acc1, current_v, current_a, horizon)
                planned_traj, lat1, yaw1, v1 = motion_skill_model(lon1, lat1, yaw1, v1, acc1, current_v, current_a, horizon)
                planned_traj = torch.tensor(planned_traj[1:, :3]).unsqueeze(0)
                traj_eval_from_planner[i, modal_id, :] = planned_traj
                pi_eval_from_planner[i, modal_id] = 1 / num_modality


                # traj = traj_eval[i, modal_id][:, :2].cpu().numpy()
                scenario_id = data['scenario_id'][i]
                cur_heading_on_map = data['agent']['heading'][eval_mask][i,50:51].item()
                cur_pos_on_map = data['agent']['position'][eval_mask][i,50:51,:2].cpu().numpy()
                # check_out_of_map(scenario_id, data, gt_eval[i,:,:2].cpu().numpy(), cur_heading_on_map, cur_pos_on_map, visualize=True)
                # out_of_map = check_out_of_map(scenario_id, data, planned_traj[0,:,:2].cpu().numpy(), cur_heading_on_map, cur_pos_on_map, visualize=True)
                # check_out_of_map(scenario_id, data, traj_eval[i,0,:,:2].cpu().numpy(), cur_heading_on_map, cur_pos_on_map, visualize=True)
            # import pdb; pdb.set_trace()


            # import pdb; pdb.set_trace()

            if ood_flag[i]:
                traj_eval_ood_switch[i] = traj_eval_from_planner[i]
                pi_eval_ood_switch[i] = pi_eval_from_planner[i]
            else:
                traj_eval_ood_switch[i] = traj_eval[i]
                pi_eval_ood_switch[i] = pi_eval[i]

        '''' testing mode '''
        test_ood_switch = True  # switch to planner when ood
        test_planner = False
        test_qcnet = False
        if test_ood_switch:
            pi_eval = pi_eval_ood_switch
            traj_eval = traj_eval_ood_switch
        elif test_planner:
            pi_eval = pi_eval_from_planner
            traj_eval = traj_eval_from_planner
        elif test_qcnet:
            pi_eval = pi_eval
            traj_eval = traj_eval

        ''' save target states, for clustering '''
        # import numpy as np
        # # lon1, lat1, yaw1, v1, acc1
        # target_v = torch.sqrt(torch.sum(torch.square(gt_eval[:, -1, :2] - gt_eval[:, -2, :2]), 1)) * 10
        # last_v = torch.sqrt(torch.sum(torch.square(gt_eval[:, -2, :2] - gt_eval[:, -3, :2]), 1)) * 10
        # target_a = (target_v - last_v) * 10
        # goal_state = torch.zeros(gt_eval.shape[0], 5)
        # goal_state[:, :3] = gt_eval[:, -1]
        # goal_state[:, 3] = target_v
        # goal_state[:, 4] = target_a
        # # import pdb; pdb.set_trace()
        # file_path = 'target_state.npy'
        # if os.path.exists(file_path):
        #     old_data = np.load(file_path)
        #     new_data = np.vstack((old_data, goal_state.numpy()))
        #     np.save(file_path, new_data)
        # else:
        #     np.save(file_path, goal_state.numpy())
        # print('eval-----------------')


        self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                          valid_mask=valid_mask_eval)
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                       valid_mask=valid_mask_eval)
        self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def test_step(self,
                  data,
                  batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()
        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], {eval_id[i]: traj_eval[i]})
            else:
                self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        return parent_parser
