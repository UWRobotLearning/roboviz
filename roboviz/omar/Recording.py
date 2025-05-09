#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
from scipy.spatial.transform import Rotation as R
import boto3
import sys 

def detect_cameras(raw_dir):
    """
    Scan the first demo_* folder in raw_dir for any `<cam>_rgb` subdirs,
    strip the `_rgb` suffix to get the camera name, and only keep it if
    a corresponding `<cam>_depth` folder also exists.
    """
    demos = sorted(glob.glob(os.path.join(raw_dir, "demo_*")))
    if not demos:
        raise RuntimeError(f"No demo_* folders found in {raw_dir}")
    sample = demos[0]
    cams = []
    for entry in os.listdir(sample):
        rgb_path = os.path.join(sample, entry)
        if os.path.isdir(rgb_path) and entry.endswith("_rgb"):
            cam = entry[:-4]  # drop the "_rgb"
            depth_path = os.path.join(sample, f"{cam}_depth")
            if os.path.isdir(depth_path):
                cams.append(cam)
    return sorted(cams)

# ------------------------ Configuration ------------------------

input_dir   = "path/to/your/raw_data"             # contains demo_0000, demo_0001, ...
output_dir  = "path/to/output/LeRobot_dataset"
fps         = 20.0                                # robot’s recording rate
chunk_size  = 1000                                # episodes per chunk folder
cameras     = detect_cameras(input_dir)
DEPTH_MAX   = 1.0                                 # depth clip at 1 m
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# LeRobot default path patterns (used in meta/info.json)
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH  = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
# ----------------------------------------------------------------
def make_dirs():
    for sub in ("data","videos","meta"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

def extract_pose_and_gripper(prop_json):
    mat  = np.array(prop_json["gripper_matrix"], dtype=np.float32)
    pos  = mat[:3,3]
    quat = R.from_matrix(mat[:3,:3]).as_quat()
    grip = float(prop_json["gripper_open"])
    return pos, quat, grip

def write_video(frames, path, is_color=True):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h), isColor=is_color)
    if not vw.isOpened():
        raise RuntimeError(f"Could not open {path}")
    for f in frames:
        vw.write(f)
    vw.release()

def main():
    make_dirs()
    total_eps = 0
    total_frames = 0
    episode_lengths = []

    demo_dirs = sorted(glob.glob(os.path.join(input_dir, "demo_*")))
    for demo_path in demo_dirs:
        # read per-demo metadata if any (omitted here for brevity)…
        ep_idx = total_eps
        chunk_str = f"chunk-{ep_idx//chunk_size:03d}"

        # ensure per-chunk dirs
        os.makedirs(os.path.join(output_dir, "data", chunk_str), exist_ok=True)
        for cam in cameras:
            # use full feature key here:
            rgb_key   = f"observation.images.{cam}_rgb"
            depth_key = f"observation.images.{cam}_depth"
            os.makedirs(os.path.join(output_dir, "videos", chunk_str, rgb_key),   exist_ok=True)
            os.makedirs(os.path.join(output_dir, "videos", chunk_str, depth_key), exist_ok=True)

        # get all proprioception files
        p_files = sorted(glob.glob(os.path.join(demo_path, "robot_data","proprioception_*.json")))
        seqs    = sorted(int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in p_files)
        N       = len(seqs)
        episode_lengths.append(N)

        # build frame-wise lists
        episode_index = [ep_idx]*N
        frame_index   = list(range(N))
        timestamps    = [i/fps for i in frame_index]
        done_flags    = [False]*N; done_flags[-1] = True
        states, actions = [], []
        global_idx = list(range(total_frames, total_frames+N))
        total_frames += N

        for i, s in enumerate(seqs):
            pj = json.load(open(os.path.join(demo_path, f"robot_data/proprioception_{s:04d}.json")))
            pos, quat, grip = extract_pose_and_gripper(pj)
            vec = np.hstack([pos, quat, [grip]]).astype(np.float32)
            states .append(vec.tolist())
            actions.append(vec.tolist())

        # write videos into the new, full-key subfolders
        for cam in cameras:
            rgb_key   = f"observation.images.{cam}_rgb"
            depth_key = f"observation.images.{cam}_depth"
            # RGB
            rgb_dir   = os.path.join(demo_path, f"{cam}_rgb")
            rgb_frames= [cv2.imread(os.path.join(rgb_dir, f"image_{s:04d}.jpg")) for s in seqs]
            rgb_out   = os.path.join(output_dir, "videos", chunk_str, rgb_key, f"episode_{ep_idx:06d}.mp4")
            write_video(rgb_frames, rgb_out, is_color=True)
            # Depth
            depth_dir   = os.path.join(demo_path, f"{cam}_depth")
            depth_frames= []
            for s in seqs:
                d = np.load(os.path.join(depth_dir, f"depth_{s:04d}.npz"))["depth"]
                d = np.clip(d,0,DEPTH_MAX)/DEPTH_MAX
                d = (d*255).astype(np.uint8)
                depth_frames.append(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
            depth_out   = os.path.join(output_dir, "videos", chunk_str, depth_key, f"episode_{ep_idx:06d}.mp4")
            write_video(depth_frames, depth_out, is_color=True)

        # 3b) build the Parquet table
        data = {
            "episode_index":      episode_index,
            "frame_index":        frame_index,
            "timestamp":          timestamps,
            "next.done":          done_flags,
            "index":              global_idx,
            "observation.state":  states,
            "action":             actions,
            "task_index":         [0]*N,
        }
        tbl = pa.Table.from_pydict(data)
        pq.write_table(
            tbl,
            os.path.join(output_dir, "data", chunk_str, f"episode_{ep_idx:06d}.parquet")
        )

        total_eps += 1

    # ——— write meta/info.json ——————————————————————————————
    info = {
        "codebase_version": "v2.1",
        "robot_type":       "franka",
        "total_episodes":   total_eps,
        "total_frames":     total_frames,
        "total_tasks":      1,
        "total_videos":     total_eps * len(cameras) * 2,
        "total_chunks":     (total_eps + chunk_size - 1)//chunk_size,
        "chunks_size":      chunk_size,
        "fps":              fps,
        "video":            True,
        "encoding": {
            "codec":   "mp4v",
            "pix_fmt": "yuv420p"
        },
        "splits": {
            "train": f"0:{total_eps}"
        },
        "data_path":  DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features": {
            "observation.state": {"dtype":"float32","shape":[8]},
            "action":            {"dtype":"float32","shape":[8]},
            # NOTE: keys must match the folder names above
            **{ f"observation.images.{cam}_rgb":   {"dtype":"video","shape":[None,None,3],"info":{"fps":fps,"codec":"mp4v"}}  for cam in cameras },
            **{ f"observation.images.{cam}_depth": {"dtype":"video","shape":[None,None,3],"info":{"fps":fps,"codec":"mp4v","is_depth_map":True}}  for cam in cameras },
        }
    }
    os.makedirs(os.path.join(output_dir,"meta"), exist_ok=True)
    with open(os.path.join(output_dir,"meta","info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # episodes.jsonl
    with open(os.path.join(output_dir,"meta","episodes.jsonl"), "w") as f:
        for i, L in enumerate(episode_lengths):
            f.write(json.dumps({"episode_index":i,"tasks":[0],"length":L})+"\n")

    # tasks.jsonl
    with open(os.path.join(output_dir,"meta","tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index":0,"task":"default"})+"\n")

    # episodes_stats.jsonl
    with open(os.path.join(output_dir,"meta","episodes_stats.jsonl"), "w") as f:
        for i in range(total_eps):
            f.write(json.dumps({"episode_index":i,"stats":{}})+"\n")

    # stats.json
    stats = {k:{} for k in info["features"].keys()}
    with open(os.path.join(output_dir,"meta","stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Conversion complete: {total_eps} eps, {total_frames} frames → {output_dir}")


def upload_directory_to_s3(directory, bucket, prefix=""):
    """
    Recursively upload a directory to S3, preserving folder structure.
    :param directory: local path to upload
    :param bucket: S3 bucket name
    :param prefix: key prefix inside the bucket (no leading slash)
    """
    s3 = boto3.client("s3")
    for root, dirs, files in os.walk(directory):
        for fname in files:
            local_path = os.path.join(root, fname)
            # build the S3 key by stripping off the base directory
            rel_path = os.path.relpath(local_path, directory)
            s3_key   = os.path.join(prefix, rel_path).replace(os.sep, "/")
            s3.upload_file(local_path, bucket, s3_key)
            print(f"Uploaded {local_path} → s3://{bucket}/{s3_key}")

if __name__ == "__main__":
    main()
    
    # Upload to S3 --
    S3_BUCKET = "your-bucket-name"
    S3_PREFIX = "optional/prefix"  # prefix in S3 bucket (optional), this is where the files will be uploaded, otherwise
    # they will be uploaded to the root of the bucket
    
    # only upload if invoked with `upload`
    if len(sys.argv) > 1 and sys.argv[1].lower() == "upload":
        S3_BUCKET = "your-bucket-name"
        S3_PREFIX = "optional/prefix"  # or leave empty
        print(f"Starting upload of '{output_dir}' to s3://{S3_BUCKET}/{S3_PREFIX}/")
        upload_directory_to_s3(output_dir, S3_BUCKET, S3_PREFIX)
        print("Upload complete.")
    else:
        print("Skipping S3 upload (pass 'upload' to enable).")
    
    
