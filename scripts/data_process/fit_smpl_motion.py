import glob
import os
import sys
import os.path as osp
import numpy as np
import torch
import joblib
from multiprocessing import Pool
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from phc.utils.torch_humanoid_batch import Humanoid_Batch
from smpl_sim.utils.smoothing_utils import gaussian_filter_1d_batch
from easydict import EasyDict
import hydra
from omegaconf import DictConfig


def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in entry_data:
        return None
    framerate = entry_data['mocap_framerate']
    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate
    }


def process_motion(key_names, key_name_to_pkls, cfg):
    device = torch.device("cpu")
    humanoid_fk = Humanoid_Batch(cfg.robot)  # Load forward kinematics model
    num_augment_joint = len(cfg.robot.extend_config)

    # Define correspondences between humanoid and SMPL joints
    robot_joint_names_augment = humanoid_fk.body_names_augment
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path="phc/data/smpl", gender="neutral")
    shape_new, scale = joblib.load(f"phc/data/{cfg.robot.humanoid_type}/shape_optimized_v1.pkl")

    all_data = {}
    for data_key in key_names:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None:
            continue

        skip = int(amass_data['fps'] // 30)
        trans = torch.from_numpy(amass_data['trans'][::skip])
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(amass_data['pose_aa'][::skip]).float()

        if N < 10:
            print(f"{data_key} is too short.")
            continue

        with torch.no_grad():
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            root_pos = joints[:, 0:1]
            joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
        joints[..., 2] -= verts[0, :, 2].min().item()

        offset = joints[:, 0] - trans
        root_trans_offset = (trans + offset).clone()

        gt_root_rot_quat = torch.from_numpy(
            (sRot.from_rotvec(pose_aa_walk[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).float()
        gt_root_rot = torch.from_numpy(sRot.from_quat(torch_utils.calc_heading_quat(gt_root_rot_quat)).as_rotvec()).float()

        dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))
        dof_pos_new = torch.autograd.Variable(dof_pos.clone(), requires_grad=True)
        root_rot_new = torch.autograd.Variable(gt_root_rot.clone(), requires_grad=True)
        root_pos_offset = torch.autograd.Variable(torch.zeros(1, 3), requires_grad=True)
        optimizer = torch.optim.Adam([dof_pos_new, root_rot_new, root_pos_offset], lr=0.02)

        kernel_size = 5
        sigma = 0.75

        for iteration in range(cfg.get("fitting_iterations", 500)):
            pose_aa_h1_new = torch.cat(
                [root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new,
                 torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis=2)
            fk_return = humanoid_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ] + root_pos_offset)

            if num_augment_joint > 0:
                diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            else:
                diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]

            loss_g = diff.norm(dim=-1).mean() + 0.01 * torch.mean(torch.square(dof_pos_new))
            loss = loss_g
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
            dof_pos_new.data = gaussian_filter_1d_batch(
                dof_pos_new.squeeze().transpose(1, 0)[None, ],
                kernel_size, sigma
            ).transpose(2, 1)[..., None]

        pose_aa_h1_new = torch.cat(
            [root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new,
             torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis=2)
        root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()

        combined_mesh = humanoid_fk.mesh_fk(pose_aa_h1_new[:, :1].detach(), root_trans_offset_dump[None, :1].detach())
        height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()
        root_trans_offset_dump[..., 2] -= height_diff
        joints_dump = joints.numpy().copy()
        joints_dump[..., 2] -= height_diff

        data_dump = {
            "root_trans_offset": root_trans_offset_dump.squeeze().detach().numpy(),
            "pose_aa": pose_aa_h1_new.squeeze().detach().numpy(),
            "dof": dof_pos_new.squeeze().detach().numpy(),
            "root_rot": sRot.from_rotvec(root_rot_new.detach().numpy()).as_quat(),
            "smpl_joints": joints_dump,
            "fps": 30
        }
        all_data[data_key] = data_dump
    return all_data

def process_motion_wrapper(args):
    key_names, key_name_to_pkls, cfg = args
    with tqdm(total=len(key_names), desc="Processing Motion") as pbar:
        result = process_motion(key_names, key_name_to_pkls, cfg)
        pbar.update(len(key_names))
    return result

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    if "amass_root" not in cfg:
        raise ValueError("amass_root is not specified in the config")

    all_pkls = glob.glob(f"{cfg.amass_root}/**/*.npz", recursive=True)
    split_len = len(cfg.amass_root.split("/")) + 1
    key_name_to_pkls = {
        "0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls
    }
    key_names = list(key_name_to_pkls.keys())

    if not cfg.get("fit_all", False):
        key_names = ["0-Transitions_mocap_mazen_c3d_dance_stand_poses"]

    num_jobs = min(100, len(key_names))
    chunk_size = max(1, len(key_names) // num_jobs)
    job_args = [(key_names[i:i + chunk_size], key_name_to_pkls, cfg) for i in range(0, len(key_names), chunk_size)]

    all_data = {}
    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        futures = {executor.submit(process_motion, *args): args for args in job_args}
        for future in tqdm(as_completed(futures), total=len(job_args), desc="Processing Chunks"):
            data_dict = future.result()  # 获取结果
            all_data.update(data_dict)

    if len(all_data) == 1:
        data_key = list(all_data.keys())[0]
        os.makedirs(f"phc/data/{cfg.robot.humanoid_type}/v1/singles", exist_ok=True)
        joblib.dump(all_data, f"phc/data/{cfg.robot.humanoid_type}/v1/singles/{data_key}.pkl")
    else:
        os.makedirs(f"phc/data/{cfg.robot.humanoid_type}/v1/", exist_ok=True)
        joblib.dump(all_data, f"phc/data/{cfg.robot.humanoid_type}/v1/amass_all.pkl")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()