import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from data.extract_frames import extract_frames
from data.vstrong_dataset import ResizeSquare
from models.vstrong_lit import VStrongLit

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _draw_header(canvas: np.ndarray, title: str, left_label: str, right_label: str) -> np.ndarray:
    h, w = canvas.shape[:2]
    bar_h = max(60, int(h * 0.08))
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[bar_h:, :, :] = canvas

    cv2.putText(
        out,
        title,
        (20, int(bar_h * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    half_w = w // 2
    cv2.putText(
        out,
        left_label,
        (20, bar_h + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        right_label,
        (half_w + 20, bar_h + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Inference-only video renderer: video -> extract frames -> V-STRONG overlay side-by-side."
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--title", type=str, default="V-STRONG Inference")
    parser.add_argument("--fps", type=float, default=0.0, help="Extraction FPS; <=0 means every frame.")
    parser.add_argument("--output_fps", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output_height", type=int, default=None)
    parser.add_argument("--keep_frames", action="store_true")
    parser.add_argument("--tmp_frames_dir", type=str, default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    ckpt_path = Path(args.ckpt_path)
    output_path = Path(args.output)
    if not video_path.exists():
        raise RuntimeError(f"Video not found: {video_path}")
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    print("Step 1/3: loading model")
    model = VStrongLit.load_from_checkpoint(str(ckpt_path), strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    transform = ResizeSquare(int(getattr(model, "img_size", getattr(model, "sam_img_size", 1024))))
    if not bool(model.traversability_initialized.item()):
        raise RuntimeError(
            "Checkpoint does not contain an initialized traversability vector. "
            "Retrain with the updated code to enable paper-style image-only inference."
        )

    if args.tmp_frames_dir:
        frames_dir = Path(args.tmp_frames_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        frames_dir = Path("/tmp") / "vstrong_infer_only" / f"{video_path.stem}_{stamp}" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("Step 2/3: extracting frames")
    metadata = extract_frames(str(video_path), str(frames_dir), target_fps=float(args.fps))
    effective_fps = float(metadata.get("effective_fps", 5.0))
    out_fps = float(args.output_fps) if args.output_fps is not None else effective_fps

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if args.max_frames is not None:
        frame_paths = frame_paths[: int(args.max_frames)]
    if len(frame_paths) == 0:
        raise RuntimeError("No frames extracted.")

    first_bgr = cv2.imread(str(frame_paths[0]))
    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    resized_first = transform.apply_image(first_rgb)
    h0, w0 = resized_first.shape[:2]

    if args.output_height is None:
        h_out = h0
        w_out = w0
    else:
        h_out = int(args.output_height)
        scale = h_out / float(h0)
        w_out = int(round(w0 * scale))

    canvas_w = 2 * w_out
    canvas_h = h_out
    header_h = max(60, int(canvas_h * 0.08))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (canvas_w, canvas_h + header_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {output_path}")

    print("Step 3/3: running inference + rendering video")
    iterable = tqdm(frame_paths, desc="Inference render", unit="frame") if tqdm is not None else frame_paths
    wrote = 0
    for fp in iterable:
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = transform.apply_image(rgb)

        emb = model._encode_images([resized_rgb]).to(device)
        z_map = model.proj(emb)[0]
        viz = model._make_viz_with_traversability_vector(resized_rgb=resized_rgb, z_map=z_map)

        left_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        right_bgr = viz.overlay_bgr
        if (left_bgr.shape[0], left_bgr.shape[1]) != (h_out, w_out):
            left_bgr = cv2.resize(left_bgr, (w_out, h_out), interpolation=cv2.INTER_AREA)
        if (right_bgr.shape[0], right_bgr.shape[1]) != (h_out, w_out):
            right_bgr = cv2.resize(right_bgr, (w_out, h_out), interpolation=cv2.INTER_AREA)

        canvas = np.concatenate([left_bgr, right_bgr], axis=1)
        merged = _draw_header(canvas, args.title, "RGB", "Traversability overlay")
        writer.write(merged)
        wrote += 1
        if tqdm is not None and hasattr(iterable, "set_postfix_str"):
            iterable.set_postfix_str(f"wrote={wrote}")

    writer.release()
    print(f"Done. wrote={wrote}, output={output_path}")

    if (not args.keep_frames) and frames_dir.exists():
        import shutil

        shutil.rmtree(frames_dir.parent if frames_dir.name == "frames" else frames_dir, ignore_errors=True)
        print("Cleaned temporary frames")


if __name__ == "__main__":
    main()
