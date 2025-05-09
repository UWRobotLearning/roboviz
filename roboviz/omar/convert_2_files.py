#!/usr/bin/env python3
"""
Convert *two* HDF5 demos files into a single LeRobot dataset:
 - Parquet tables under data/chunk-*/episode_*.parquet
 - MP4 videos under videos/chunk-*
 - Complete meta/ folder with info.json, tasks.jsonl, episodes.jsonl,
   episodes_stats.jsonl, and stats.json
"""

import math
import json
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
import jsonlines
from pathlib import Path

# ---------- User config: two input HDF5s ----------
hdf5_paths   = [
    "/Users/omarabdelaziz/Downloads/robodata/expert_lampshade2_demos.hdf5",
    "/Users/omarabdelaziz/Downloads/robodata/play_pushing.hdf5",
]
out_root     = "/Users/omarabdelaziz/Downloads/robodata/Combined_LeRobot_dataset"
fps          = 20.0
chunk_size   = 1000
episodes_grp = "data"

# camera fields: key -> HDF5 subpath
cams = {
    "camera_front_real": "obs/front_image",
    "camera_wrist_real": "obs/wrist_image",
}

# LeRobot default path patterns (for info.json)
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH   = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
# ---------------------------------------------------------------------

def make_dirs():
    for sub in ("data", "videos", "meta"):
        Path(out_root, sub).mkdir(parents=True, exist_ok=True)

def write_video(frames, path, is_color=True):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=is_color)
    if not vw.isOpened():
        raise RuntimeError(f"Could not open {path} for writing")
    for f in frames:
        vw.write(f)
    vw.release()

def main():
    make_dirs()

    # 1) Infer feature dims from first HDF5
    with h5py.File(hdf5_paths[0], "r") as f0:
        first_demo = sorted(f0[episodes_grp].keys())[0]
        grp0       = f0[f"{episodes_grp}/{first_demo}"]
        state_dim  = grp0["obs/states"].shape[1]
        action_dim = grp0["actions"].shape[1]
        H, W, C    = grp0[next(iter(cams.values()))][0].shape

    # 2) Build features dict
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
        "action":            {"dtype": "float32", "shape": [action_dim]},
        "reward":            {"dtype": "float32", "shape": []},
        "next.done":         {"dtype": "bool",    "shape": []},
    }
    for cam_key in cams:
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": [H, W, C],
            "info": {"fps": fps, "codec": "mp4v", "pix_fmt": "yuv420p"},
        }

    splits         = {"train": [0, 0]}
    total_eps      = 0
    total_frames   = 0
    episode_lengths = []

    # 3) Process each HDF5
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(f[episodes_grp].keys())
            for demo in demos:
                grp = f[f"{episodes_grp}/{demo}"]
                T   = grp["obs/states"].shape[0]

                # prepare chunk folder
                chunk_str = f"chunk-{total_eps // chunk_size:03d}"
                Path(out_root, "data", chunk_str).mkdir(parents=True, exist_ok=True)
                for cam_key in cams:
                    video_key = f"observation.images.{cam_key}"
                    Path(out_root, "videos", chunk_str, video_key).mkdir(parents=True, exist_ok=True)

                # read arrays
                states_arr  = grp["obs/states"][:] .astype(np.float32)
                actions_arr = grp["actions"][:]    .astype(np.float32)
                rewards_arr = grp["rewards"][:]    .astype(np.float32)
                dones_arr   = grp["dones"][:]      .astype(bool)

                # build indices & timestamps
                ep_idx_list = [total_eps] * T
                frame_idx   = list(range(T))
                timestamps  = [i / fps for i in frame_idx]
                global_idxs = list(range(total_frames, total_frames + T))
                episode_lengths.append(T)
                total_frames += T

                # 3a) write videos
                for cam_key, h5path in cams.items():
                    video_key = f"observation.images.{cam_key}"
                    rgb_frames = [grp[h5path][i] for i in range(T)]
                    out_vid    = Path(out_root, "videos", chunk_str, video_key,
                                      f"episode_{total_eps:06d}.mp4")
                    write_video(rgb_frames, out_vid, is_color=True)

                # 3b) write Parquet
                data = {
                    "episode_index":      ep_idx_list,
                    "frame_index":        frame_idx,
                    "timestamp":          timestamps,
                    "next.done":          dones_arr.tolist(),
                    "reward":             rewards_arr.tolist(),
                    "observation.state":  states_arr.tolist(),
                    "action":             actions_arr.tolist(),
                    "task_index":         [0] * T,
                    "index":              global_idxs,
                }
                tbl = pa.Table.from_pydict(data)
                pq.write_table(
                    tbl,
                    Path(out_root, "data", chunk_str, f"episode_{total_eps:06d}.parquet")
                )

                total_eps += 1

    # 4) finalize splits & chunks
    splits["train"] = [0, total_eps]
    total_chunks   = math.ceil(total_eps / chunk_size)

    # 5) write meta/info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type":       "franka",
        "total_episodes":   total_eps,
        "total_frames":     total_frames,
        "total_tasks":      1,
        "total_videos":     total_eps * len(cams),
        "total_chunks":     total_chunks,
        "chunks_size":      chunk_size,
        "fps":              fps,
        "video":            True,
        "encoding": {
            "codec":   "mp4v",
            "pix_fmt": "yuv420p"
        },
        "splits":     splits,
        "data_path":  DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features":   features
    }
    with open(Path(out_root, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # 6) write tasks, episodes, episodes_stats
    with jsonlines.open(Path(out_root, "meta", "tasks.jsonl"), "w") as w:
        w.write({"task_index": 0, "task": "default"})
    with jsonlines.open(Path(out_root, "meta", "episodes.jsonl"), "w") as w:
        for idx, length in enumerate(episode_lengths):
            w.write({"episode_index": idx, "tasks": [0], "length": length})
    with jsonlines.open(Path(out_root, "meta", "episodes_stats.jsonl"), "w") as w:
        for idx in range(total_eps):
            w.write({"episode_index": idx, "stats": {}})

    # 7) stub stats.json
    stub = {feat: {} for feat in features.keys()}
    with open(Path(out_root, "meta", "stats.json"), "w") as f:
        json.dump(stub, f, indent=2)

    print(f"Done: wrote {total_eps} episodes, {total_frames} frames → {out_root}")

if __name__ == "__main__":
    main()