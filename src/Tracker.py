import time
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from src.utils import get_tensor_from_camera, get_camera_from_tensor
from src.Datasets import get_datasetloader
from src.profiling import nvtx_range


class Tracker(object):
    def __init__(self, cfg, args, slam):

        self.cfg = cfg
        self.args = args
        self.output = slam.output
        self.verbose = slam.verbose
        self.scale = cfg['scale']
        self.sleepTime = cfg['sleepTime']
        self.logger = slam.logger
        self.debug = cfg["debug"]["flag"]
        self.checkpoint = cfg["debug"]["checkpoint"]
        self.showTrackerLoss = cfg["debug"]["showTrackerLoss"]
        self.last_best = cfg["tracking"]["last_best"]
        self.max_n_models = cfg["mapping"]["max_n_models"]
        self.lr_T = cfg['tracking']['lr_T']
        self.lr_R = cfg['tracking']['lr_R']
        self.device = cfg["device"]
        self.every_frame = cfg["mapping"]["every_frame"]
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']
        self.num_cam_iters = cfg['tracking']['iters']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.tracking_pixels_bg = cfg['tracking']['pixels_bg']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.idx = slam.idx
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.estimate_relative_c2w_list = slam.estimate_relative_c2w_list
        self.gt_c2w_list = slam.gt_c2w_list
        self.mapping_first_frame = slam.mapping_first_frame
        self.mapping_idx = slam.mapping_idx
        self.vis_dict = slam.vis_dict
        self.mapping_cnt = slam.mapping_cnt
        self.iteration_timer = slam.iteration_timer
        self.pe_buffer = None
        self.pe_param = None
        self.pe_model = None
        self.fc_buffer = None
        self.fc_param = None
        self.fc_model = None
        self.optimize_inst_ids = []
        self.fc_models = []
        self.pe_models = []
        self.prev_mapping_idx = -1
        self.frame_reader = slam.frame_reader
        self.frame_loader = get_datasetloader(self.frame_reader)
        self.frameloader_iterator = iter(self.frame_loader)
        self.n_img = len(self.frame_loader)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def optimize_cam_in_batch(
        self,
        color,
        depth,
        features,
        optimizer,
        gt_camera_tensor,
        quad,
        T,
        frame_id,
    ):
        with nvtx_range("initialize_tracking_metrics_tr"):
            camera_tensor = torch.cat([quad, T], 0).detach()
            initial_loss_camera_tensor = torch.abs(
                camera_tensor - gt_camera_tensor.to(self.device)
            ).mean().item()
            candidate_cam_tensor = None
            current_min_loss = 10000000000.

        # EC-SLAM samples once and reuses the same pixels for every pose iteration.
        with nvtx_range("select_samples_tr"):
            scene = self.vis_dict[0]
            bg_rgb, bg_dir, bg_depth = scene.get_tracking_samples(
                self.tracking_pixels_bg, color, depth, features
            )
        with nvtx_range("initialize_pose_optimizer_tr"):
            optimizer.zero_grad()

        for cam_iter in range(self.num_cam_iters):
            with self.iteration_timer.stage(
                frame_id,
                "tracking",
                cam_iter,
                "full_iteration_cuda_ms",
            ), nvtx_range("TR_ITER_PROFILE_TOTAL"):
                with nvtx_range("pos_matrix_from_tensor_tr"):
                    camera_tensor = torch.cat([quad, T], 0)
                    c2w = get_camera_from_tensor(camera_tensor)

                with nvtx_range("Tensor_ops_s&d_tr"):
                    bg_data_idx = slice(
                        0 * self.tracking_pixels_bg,
                        (0 + 1) * self.tracking_pixels_bg,
                    )
                    bg_origin = c2w[:, 3]
                    bg_dirW = (
                        c2w[None, :, :3] @ bg_dir[bg_data_idx, :, None]
                    ).squeeze()
                    temp_s = bg_rgb[bg_data_idx].detach()
                    temp_d = bg_depth[bg_data_idx].detach()[..., None]

                # Model.forward computes both predictions and the weighted loss in EC-SLAM.
                with self.iteration_timer.stage(
                    frame_id,
                    "tracking",
                    cam_iter,
                    "forward_cuda_ms",
                ), nvtx_range("forward_tr"):
                    bg_loss = scene.vMapModel.forward(
                        bg_origin.repeat(bg_dirW.shape[0], 1),
                        bg_dirW,
                        temp_s,
                        temp_d,
                        tracker=True,
                        smooth=False,
                    )
                    batch_loss = bg_loss

                with self.iteration_timer.stage(
                    frame_id,
                    "tracking",
                    cam_iter,
                    "backward_cuda_ms",
                ), nvtx_range("backward_tr"):
                    batch_loss.backward()
                with nvtx_range("pose_optimizer_tr"):
                    optimizer.step()
                    optimizer.zero_grad()

                with nvtx_range("select_best_pose_tr"):
                    current_loss = batch_loss.item()
                    if cam_iter == 0:
                        initial_loss = current_loss
                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(self.device) - camera_tensor
                    ).mean().item()
                    if self.last_best:
                        candidate_cam_tensor = camera_tensor.clone().detach()
                    elif current_loss < current_min_loss:
                        current_min_loss = current_loss
                        candidate_cam_tensor = camera_tensor.clone().detach()
                    if self.showTrackerLoss and self.debug:
                        print(cam_iter, current_loss, loss_camera_tensor)
                    if self.verbose and cam_iter == self.num_cam_iters - 1:
                        print(
                            f'Re-rendering loss: {initial_loss:.2f}->{current_loss:.2f} '
                            + 'camera tensor error: '
                            + f'{initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}'
                        )

        with nvtx_range("finalize_best_pose_tr"):
            best_loss_camera_tensor = torch.abs(
                gt_camera_tensor.to(self.device) - candidate_cam_tensor
            ).mean().item()
        if self.verbose:
            print(f'Best camera loss:{best_loss_camera_tensor:.4f}')
        return candidate_cam_tensor

    def run(self):
        pbar = tqdm(self.frame_loader)
        for sample in pbar:
            # The parent lives in the tracking worker so all child ranges share
            # the same thread-local NVTX stack.
            with nvtx_range("Tracking"):
                idx = sample["frame_id"]
                wait_for_mapping = idx > 0 and (
                    idx % self.every_frame == 1 or self.every_frame == 1
                )

                # During timing runs, wait before enqueueing the next frame's
                # H2D copies. Otherwise those copies can land between the
                # mapper's CUDA events and contaminate its BA measurements.
                if wait_for_mapping and self.iteration_timer.enabled:
                    with nvtx_range("wait_for_mapping_tr"):
                        while self.mapping_idx[0] != idx - 1:
                            time.sleep(self.sleepTime)

                with nvtx_range("frame_data_to_gpu_tr"):
                    gt_color = sample["image"].to(self.device, torch.float32)
                    gt_depth = sample["depth"].to(self.device)
                    gt_c2w = sample["T"].to(self.device)
                    features = sample["features"]

                if wait_for_mapping and not self.iteration_timer.enabled:
                    with nvtx_range("wait_for_mapping_tr"):
                        while self.mapping_idx[0] != idx - 1:
                            time.sleep(self.sleepTime)

                if wait_for_mapping:
                    pre_c2w = self.estimate_c2w_list[idx - 1].to(self.device)

                with nvtx_range("initialize_pose_tr"):
                    if idx == 0:
                        c2w = gt_c2w.float()
                        if self.verbose:
                            print("Using ground truth to set current frame's pose")
                    else:
                        gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                        if self.const_speed_assumption and idx >= 2:
                            pre_c2w = pre_c2w.float()
                            delta = pre_c2w @ self.estimate_c2w_list[idx - 2].to(
                                self.device
                            ).float().inverse()
                            estimated_new_cam_c2w = delta @ pre_c2w
                        else:
                            estimated_new_cam_c2w = pre_c2w
                        camera_tensor = get_tensor_from_camera(
                            estimated_new_cam_c2w.detach()
                        )
                        camera_tensor = camera_tensor.to(self.device).detach()
                        T = Variable(camera_tensor[-3:], requires_grad=True)
                        quad = Variable(camera_tensor[:4], requires_grad=True)
                        poseOptimizer = torch.optim.Adam(
                            [
                                {'params': [T], 'lr': self.lr_T},
                                {'params': [quad], 'lr': self.lr_R},
                            ]
                        )

                if idx != 0:
                    candidate_cam_tensor = self.optimize_cam_in_batch(
                        gt_color,
                        gt_depth,
                        features,
                        poseOptimizer,
                        gt_camera_tensor,
                        quad,
                        T,
                        idx,
                    )
                    with nvtx_range("c2w_from_matrix_tr"):
                        bottom = torch.from_numpy(
                            np.array([0, 0, 0, 1.]).reshape([1, 4])
                        ).type(torch.float32).to(self.device)
                        c2w = get_camera_from_tensor(
                            candidate_cam_tensor.clone().detach()
                        )
                        c2w = torch.cat([c2w, bottom], dim=0)

                with nvtx_range("save_pose_frame_tr"):
                    self.estimate_c2w_list[idx] = c2w.clone().cpu()
                    if idx % self.every_frame == 0:
                        self.estimate_relative_c2w_list[idx] = c2w.clone().cpu()
                    else:
                        kf_id = int(idx // self.every_frame) * self.every_frame
                        self.estimate_relative_c2w_list[idx] = (
                            self.estimate_relative_c2w_list[kf_id].inverse()
                            @ c2w.clone().cpu()
                        )
                    self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
                    pre_c2w = c2w.clone()
                    self.idx[0] = idx
                    pbar.set_description("Tracking Frame {}".format(idx))

                if idx == self.n_img - 1:
                    with nvtx_range("wait_for_final_mapping_tr"):
                        while self.mapping_idx[0] != self.n_img - 1:
                            time.sleep(0.1)
