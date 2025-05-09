
"""
Convert an HDF5 demos file into LeRobot format:
 - Parquet tables under data/chunk-*/episode_*.parquet
 - MP4 videos for front and wrist cams under videos/chunk-*
 - Complete meta/ folder with info.json, tasks.jsonl,
   episodes.jsonl, episodes_stats.jsonl, and stats.json
"""

import os
import math
import json
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
import jsonlines
from pathlib import Path

# ---------- User config ----------
hdf5_path   = "/Users/omarabdelaziz/Downloads/robodata/expert_lampshade2_demos.hdf5"
out_root    = "/Users/omarabdelaziz/Downloads/robodata/Exeprt_LeRobot_dataset"
fps         = 20.0
chunk_size  = 1000
episodes_grp = "data"
mask_ds      = "mask/train"

# camera fields: key becomes part of feature name, value is HDF5 subpath
cams = {
    "camera_front_real": "obs/front_image",
    "camera_wrist_real": "obs/wrist_image"
}

# LeRobot default path patterns
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH  = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
# ----------------------------------------------------------------

def make_dirs():
    Path(out_root, "data").mkdir(parents=True, exist_ok=True)
    Path(out_root, "videos").mkdir(parents=True, exist_ok=True)
    Path(out_root, "meta").mkdir(parents=True, exist_ok=True)

def write_video(frames, path):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=True)
    if not vw.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for f in frames:
        vw.write(f)
    vw.release()

def main():
    make_dirs()

    # 1) Infer feature dimensions
    with h5py.File(hdf5_path, "r") as f:
        demos      = sorted(f[episodes_grp].keys())
        grp0       = f[f"{episodes_grp}/{demos[0]}"]
        state_dim  = grp0["obs/states"].shape[1]
        action_dim = grp0["actions"].shape[1]
        H, W, C    = grp0[next(iter(cams.values()))][0].shape

    # 2) Build the LeRobot features dict
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
            "names": ["height", "width", "channels"],
            "info": {"fps": fps, "codec": "mp4v", "pix_fmt": "yuv420p"}
        }

    splits = {"train": [0, 0]}
    total_eps = 0
    total_frames = 0
    episode_lengths = []

    # 3) Write videos and Parquet per episode
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f[episodes_grp].keys())
        for demo in demos:
            grp = f[f"{episodes_grp}/{demo}"]
            T   = grp["obs/states"].shape[0]

            # prepare episode data arrays
            states  = grp["obs/states"][:] .astype(np.float32)
            actions = grp["actions"][:]    .astype(np.float32)
            rewards = grp["rewards"][:]    .astype(np.float32)
            dones   = grp["dones"][:]      .astype(bool)

            # chunk folder name
            chunk_str = f"chunk-{total_eps // chunk_size:03d}"

            # 3a) Encode and write videos for each camera
            for cam_key, h5path in cams.items():
                video_key = f"observation.images.{cam_key}"
                vid_dir = Path(out_root, "videos", chunk_str, video_key)
                vid_dir.mkdir(parents=True, exist_ok=True)
                frames = [grp[h5path][i] for i in range(T)]
                vid_path = vid_dir / f"episode_{total_eps:06d}.mp4"
                write_video(frames, str(vid_path))

            # 3b) Build and write the Parquet table
            ep_idx_list = [total_eps] * T
            frame_idx   = list(range(T))
            timestamps  = [i / fps for i in frame_idx]
            data = {
                "episode_index":      ep_idx_list,
                "frame_index":        frame_idx,
                "timestamp":          timestamps,
                "next.done":          dones.tolist(),
                "reward":             rewards.tolist(),
                "observation.state":  states.tolist(),
                "action":             actions.tolist(),
                # ← add task_index so loader can map into tasks.jsonl
                "task_index":         [0] * T,
            }
            tbl = pa.Table.from_pydict(data)
            data_dir = Path(out_root, "data", chunk_str)
            data_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(tbl, str(data_dir / f"episode_{total_eps:06d}.parquet"))

            episode_lengths.append(T)
            total_frames += T
            total_eps    += 1

    # 4) Finalize splits & chunk count
    splits["train"] = [0, total_eps]
    total_chunks   = math.ceil(total_eps / chunk_size)

    # 5) Write meta/info.json
    meta_dir = Path(out_root, "meta")
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
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # 6) Write tasks.jsonl, episodes.jsonl, episodes_stats.jsonl
    with jsonlines.open(meta_dir / "tasks.jsonl", mode="w") as w:
        w.write({"task_index": 0, "task": "default"})
    with jsonlines.open(meta_dir / "episodes.jsonl", mode="w") as w:
        for idx, length in enumerate(episode_lengths):
            w.write({"episode_index": idx, "tasks": [0], "length": length})
    with jsonlines.open(meta_dir / "episodes_stats.jsonl", mode="w") as w:
        for idx in range(len(episode_lengths)):
            w.write({"episode_index": idx, "stats": {}})

    # 7) Stub out stats.json (loader will recompute if empty)
    stub = {feat: {} for feat in features.keys()}
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stub, f, indent=2)

    print(f"Done: wrote {total_eps} episodes, {total_frames} frames → {out_root}")

if __name__ == "__main__":
    main()