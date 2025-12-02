from __future__ import annotations

import os

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from utils.visualiser import Visualiser
from models.ctcm import (
    ctcm_model_block_1_2, ctcm_model_block_3, 
    rails2track, CorrectionEnlargingBlocks
)
from models.bisenet import BiSeNetV2, sem_cc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluatorConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    target_size: Tuple[int, int] = (1280, 720)
    model_inference_size: Tuple[int, int] = (512, 1024)
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
# Image Evaluator
# ---------------------------------------------------------------------------
class ImageEvaluator:
    def __init__(self, config: EvaluatorConfig) -> None:
        self.cfg = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize models
        self.bisenet = self._load_model(BiSeNetV2(), "models/bisenet/bisenet.pth")
        self.postprocess_semcc = sem_cc()
        self.ctcm_block_1_2 = self._load_model(ctcm_model_block_1_2(
            block1_weights="models/ctcm/block1_weights.pth",
            block2_weights="models/ctcm/block2_weights.pth"
        ))
        self.ctcm_block_3 = self._load_model(ctcm_model_block_3(
            block3_weights="models/ctcm/block3_weights.pth"
        ))

        self.ceb = CorrectionEnlargingBlocks(self.device)

        self.to_tensor = transforms.ToTensor()
        self.logger.info("Models initialized on %s", self.device)

    def _load_model(self, model: torch.nn.Module, weights_path: str | None = None) -> torch.nn.Module:
        if weights_path:
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model.to(self.device)
        model.eval()
        return model

    # -----------------------------------------------------------------------
    # Main function
    # -----------------------------------------------------------------------
    def run(self, image_path: Path, output_path: Path) -> None:
        """Run full image evaluation pipeline."""

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Cannot open image: {image_path}")

        frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.cfg.target_size)
        image_tensor = self.to_tensor(frame_resized).unsqueeze(0).to(self.device)

        # BiSeNet + sem-cc
        bisenet_mask = self._process_bisenet(image_tensor)
        bisenet_np = bisenet_mask.detach().cpu().numpy().astype(np.uint8)[0]
        bisenet_semcc = torch.from_numpy(self.postprocess_semcc.process(bisenet_np))

        # CTCM processing
        ctcm_mask = self._process_ctcm(image_tensor)
        ctcm_track = rails2track().forward(ctcm_mask.squeeze(0).long().detach()).unsqueeze(0)

        # Visualization
        bisenet_vis = torch.take(torch.tensor([2, 1, 0]), bisenet_mask.squeeze(0).long())
        bisenet_semcc_vis = torch.take(torch.tensor([2, 1, 0]), bisenet_semcc.squeeze(0).long())
        ctcm_vis = torch.take(torch.tensor([0, 2, 1]), ctcm_track.squeeze(0).long())

        grid = Visualiser().create_grid(
            image=frame_resized,
            data=[bisenet_vis, bisenet_semcc_vis, ctcm_vis],
            names=["BiSeNetV2", "BiSeNetV2 (sem-cc)", "CTCM"],
            show_names=False
        )
        cv2.imwrite(str(output_path), grid)

        if self.cfg.debug_preview:
            cv2.imshow("result", grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # -----------------------------------------------------------------------
    # Model processing
    # -----------------------------------------------------------------------
    def _process_bisenet(self, image: torch.Tensor) -> torch.Tensor:
        img = F.interpolate(image, size=self.cfg.model_inference_size, mode="bilinear", antialias=True)
        with torch.no_grad():
            mask = torch.argmax(self.bisenet(img)[0], dim=1)
        return mask

    def _process_ctcm(self, image: torch.Tensor) -> torch.Tensor:
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

        with torch.no_grad():
            mask_track = self.ctcm_block_1_2(img)
        ctx_mask = ccl(mask_track.detach().cpu().numpy().squeeze(0)).to(self.device).unsqueeze(0)

        with torch.no_grad():
            mask_rails = self.ctcm_block_3(ctx_mask, img)

        return mask_rails


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image evaluator (BiSeNetV2 + CTCM)")
    p.add_argument("input", type=Path, help="Input directory containing images")
    p.add_argument("output", type=Path, help="Output directory for processed images")
    p.add_argument("--debug", action="store_true", help="Show preview window")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Batch processing for all images in directory
# ---------------------------------------------------------------------------
def process_directory(evaluator: ImageEvaluator, input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"Input path '{input_dir}' is not a directory or does not exist.")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in image_exts])

    if not image_paths:
        print(f"No image files found in {input_dir}")
        return

    pbar = tqdm(image_paths)
    for img_path in pbar:
        out_path = output_dir / f"{img_path.stem}_result.png"
        pbar.set_description(f"Processing {img_path.name} → {out_path.name}")

        if not os.path.exists(out_path):
        #if img_path.stem == "KZ49g":
            evaluator.run(img_path, out_path)


def main() -> None:
    args = parse_args()
    cfg = EvaluatorConfig(debug_preview=args.debug)
    setup_logging(cfg.logging_level)

    evaluator = ImageEvaluator(cfg)
    process_directory(evaluator, args.input, args.output)


if __name__ == "__main__":
    main()

