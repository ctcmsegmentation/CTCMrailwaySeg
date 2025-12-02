from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from utils.visualiser import Visualiser
from models.ctcm import (
    ctcm_model_block_1_2,
    ctcm_model_block_3,
    rails2track,
    CorrectionEnlargingBlocks
)
from models.bisenet import BiSeNetV2, sem_cc


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluatorConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_fps_divisor: int = 5
    input_size: Tuple[int, int] = (1280, 720)
    output_size: Tuple[int, int] = (int(1280*2), int(720*2))
    model_inference_size: Tuple[int, int] = (512, 1024)
    cardinality_min: float = 5711.63
    cardinality_max: float = 12025.27
    correction_delta_threshold: float = 115.0
    correction_delay_frames: int = 10
    max_frames: Optional[int] = None
    timing_csv: Optional[Path] = Path("times.csv")
    debug_preview: bool = False
    logging_level: int = logging.INFO


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Main Evaluator Class
# ---------------------------------------------------------------------------
class VideoEvaluator:
    def __init__(self, config: EvaluatorConfig) -> None:
        self.cfg = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize models
        self.bisenet = self._load_model(BiSeNetV2(), "models/bisenet/bisenet.pth")
        self.postprocess_semcc = sem_cc()
        self.ctcm_block_1_2 = self._load_model(ctcm_model_block_1_2(
            block1_weights="models/ctcm/block1_weights.pth", block2_weights="models/ctcm/block2_weights.pth"
        ))
        self.ctcm_block_3 = self._load_model(ctcm_model_block_3(block3_weights="models/ctcm/block3_weights.pth"))
        self.ceb = CorrectionEnlargingBlocks(self.device)

        self.to_tensor = transforms.ToTensor()
        self.logger.info("Models initialized on %s", self.device)

    def _load_model(self, model: torch.nn.Module, weights_path: str=None) -> torch.nn.Module:
        if weights_path:
            model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        model.to(self.device)
        model.eval()
        return model

    # -------------------------------------------------------------------
    # Main processing function
    # -------------------------------------------------------------------
    def run(self, video_path: Path, output_path: Path) -> None:
        """Run full video evaluation pipeline."""
        self.logger.info("Processing video: %s", video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        out_fps = max(1, int(fps / self.cfg.output_fps_divisor))
        out_writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            out_fps, self.cfg.output_size
        )

        total_iter = (
            frame_count
            if self.cfg.max_frames is None
            else min(frame_count, self.cfg.max_frames)
        )
        pbar = tqdm(total=total_iter, desc="Frames", unit="frame")

        ctcm_context: Optional[torch.Tensor] = None
        correction_delay = 0

        bisenet_time = semcc_time = ctcm_time = 0.0
        frame_counter = 0

        csv_writer = None
        if self.cfg.timing_csv:
            csv_file = open(self.cfg.timing_csv, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["bisenet", "bisenet_semcc", "ctcm"])
        else:
            csv_file = None

        # ----------------------------------------
        # Frame loop
        # ----------------------------------------
        for _ in range(total_iter):
            ret, frame_bgr = cap.read()

            pbar.update(1)
            if (pbar.n - 1) % self.cfg.output_fps_divisor != 0:
                continue

            if ret:    
                frame_counter += 1
            
                # Preprocess
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb_resized = cv2.resize(frame_rgb, self.cfg.input_size)
                image_tensor = self.to_tensor(frame_rgb_resized).unsqueeze(0).to(self.device)

                # ----------------------------------------
                # BiSeNet + sem-cc
                # ----------------------------------------
                t0 = time()
                bisenet_mask = self._process_bisenet(image_tensor)
                t1 = time()
                bisenet_np = bisenet_mask.detach().cpu().numpy().astype(np.uint8)[0]
                bisenet_semcc = torch.from_numpy(self.postprocess_semcc.process(bisenet_np))
                t2 = time()

                # ----------------------------------------
                # CTCM processing
                # ----------------------------------------
                ctcm_mask, ctcm_context = self._process_ctcm(image_tensor, ctcm_context)
                ctcm_eroded = self._erode_mask(ctcm_mask)
                ctcm_result = ctcm_eroded.clone()
                ctcm_eroded = self._erode_mask(ctcm_eroded)

                correction_delta = self.ceb.correction_policy(ctcm_eroded[0, :, :])
                cardinality = int(torch.sum(ctcm_eroded == 1).item())

                need_correction = (
                    correction_delta > self.cfg.correction_delta_threshold
                    or not (self.cfg.cardinality_min < cardinality < self.cfg.cardinality_max)
                )

                if correction_delay == 0 and need_correction:
                    correction_delay = self.cfg.correction_delay_frames
                elif correction_delay > 0:
                    correction_delay -= 1
                    if correction_delay == 0:
                        ctcm_context = None

                if ctcm_context is not None and torch.any(ctcm_mask):
                    ctcm_context = self.ceb.enlarging_policy(ctcm_mask)

                if not need_correction:
                    ctcm_track = rails2track().forward(
                        ctcm_result.squeeze(0).long().detach().cpu()
                    ).unsqueeze(0)
                else:
                    ctcm_track = ctcm_result

                t3 = time()

                if csv_writer:
                    csv_writer.writerow([t1 - t0, t2 - t1, t3 - t2])

                bisenet_time += (t1 - t0)
                semcc_time += (t2 - t0)
                ctcm_time += (t3 - t2)

                # ----------------------------------------
                # Visualisation
                # ----------------------------------------
                bisenet_vis = torch.take(torch.tensor([2, 1, 0]), bisenet_mask.squeeze(0).long().detach().cpu())
                bisenet_semcc_vis = torch.take(torch.tensor([2, 1, 0]), bisenet_semcc.squeeze(0).long().detach().cpu())
                ctcm_vis = torch.take(torch.tensor([0, 2, 1]), ctcm_track.squeeze(0).long().detach().cpu())

                grid = Visualiser().create_grid(
                    image=frame_rgb,
                    data=[bisenet_vis, bisenet_semcc_vis, ctcm_vis],
                    names=["BiSeNetV2", "BiSeNetV2 (sem-cc)", "CTCM"],
                )
                grid_resized = cv2.resize(grid, self.cfg.output_size)
                out_writer.write(grid_resized)

                if self.cfg.debug_preview:
                    cv2.imshow("result", grid_resized)
                    cv2.waitKey(0)

        # ----------------------------------------
        # Cleanup and summary
        # ----------------------------------------
        if csv_file:
            csv_file.close()

        pbar.close()
        cap.release()
        out_writer.release()

        if frame_counter > 0:
            self.logger.info("Avg BiSeNet time: %.6f s", bisenet_time / frame_counter)
            self.logger.info("Avg sem-cc time: %.6f s", semcc_time / frame_counter)
            self.logger.info("Avg CTCM time:   %.6f s", ctcm_time / frame_counter)
        else:
            self.logger.warning("No frames processed.")

    # -------------------------------------------------------------------
    # Model stages
    # -------------------------------------------------------------------
    def _process_bisenet(self, image: torch.Tensor) -> torch.Tensor:
        """Run BiSeNetV2 and return argmax mask."""
        img = F.interpolate(image, size=self.cfg.model_inference_size, mode="bilinear", antialias=True)
        with torch.no_grad():
            mask = torch.argmax(self.bisenet(img)[0], dim=1)
        return mask

    def _process_ctcm(
        self, image: torch.Tensor, ctx_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run CTCM two-block pipeline with context."""
        img = F.interpolate(image, size=self.cfg.model_inference_size, mode="bilinear", antialias=True)

        def ccl(sample: np.ndarray) -> torch.Tensor:
            sample = sample.astype(np.uint8)
            _, connected_components = cv2.connectedComponentsWithAlgorithm(
                sample, 
                connectivity=4, 
                ltype=cv2.CV_32S,
                ccltype=3
            )
            connected_components_torch = torch.from_numpy(connected_components)
            torch_sample = torch.from_numpy(sample)

            _, cc_counts = torch.unique(connected_components_torch, return_counts=True)
            topk_v, topk_i = torch.topk(cc_counts, k=min(len(cc_counts), 4))
            best = [np.inf, 0]
            for i in range(len(topk_v)):
                coords = torch.nonzero(connected_components_torch == topk_i[i], as_tuple=False)
                component = torch.where(
                    connected_components_torch == topk_i[i], 1.0, 0.0
                ).type(torch.float32)
                if torch.sum(torch_sample[coords[:, 0], coords[:, 1]]) > 0:
                    dsts = self.ceb.correction_policy(component) / torch.sum(component)
                    if dsts < best[0]:
                        best = [dsts, component]
            return best[1]

        if ctx_mask is None:
            with torch.no_grad():
                mask_track = self.ctcm_block_1_2(img)
            ctx_mask = ccl(mask_track.detach().cpu().numpy().squeeze(0)).to(self.device).unsqueeze(0)

        with torch.no_grad():
            mask_rails = self.ctcm_block_3(ctx_mask, img)

        return mask_rails, ctx_mask

    @staticmethod
    def _erode_mask(mask: torch.Tensor) -> torch.Tensor:
        """Erode mask with 3x3 kernel (same logic as original)."""
        if mask.ndim == 3:
            mask_in = mask.unsqueeze(1).float()
        else:
            mask_in = mask.float()
        kernel = torch.ones((1, 1, 3, 3), device=mask.device)
        conv = F.conv2d(mask_in, kernel, padding=1)
        return (conv == 9).squeeze(1).to(mask.dtype)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video evaluator (BiSeNetV2 + CTCM)")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--debug", action="store_true", help="Show preview windows")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--csv", type=Path, default=Path("times.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvaluatorConfig(
        debug_preview=args.debug,
        timing_csv=args.csv,
        max_frames=(args.max_frames if args.max_frames > 0 else None),
    )
    setup_logging(cfg.logging_level)
    evaluator = VideoEvaluator(cfg)
    evaluator.run(args.input, args.output)


if __name__ == "__main__":
    main()
