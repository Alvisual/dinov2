from enum import StrEnum
from typing import Annotated

import numpy as np
import torch
import torchvision.transforms as T
from cv2.typing import MatLike
from numpy.typing import NDArray
from torch import nn, Tensor


SHAPE2D = Annotated[tuple[int, int], "(2,)", "(H, W) General 2D Shape"]
BCHWF32 = Annotated[Tensor, "(B, C=RGB, H, W)", "Float32 Batch Image Tensor"]
BHWCUI8 = Annotated[NDArray[np.uint8] | MatLike, "(B, H, W, C=BGR)", "Uint8 Batch Image Array"]  # fmt: skip
BGRUI8 = Annotated[NDArray[np.uint8] | MatLike, "(H, W, C)", "Uint8 BGR Image Array"]
BTOKEN = Annotated[NDArray[np.float32], "(B, N, F)", "Float32 Batch (Class, Register, Patch) Token"]  # fmt: skip
BCTOKEN = Annotated[NDArray[np.float32], "(B, F)", "Float32 Batch Class Token"]
BRTOKEN = Annotated[NDArray[np.float32], "(B, N, F)", "Float32 Batch Register Token"]
BPTOKEN = Annotated[NDArray[np.float32], "(B, PH*PW, F)", "Float32 Batch Patch Token"]


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Arch(StrEnum):
    SMALL = "vits14"
    BASE = "vitb14"
    LARGE = "vitl14"
    GIANT = "vitg14"


class NaiveDino(nn.Module):
    def __init__(
        self,
        image_shape: SHAPE2D,
        backbone_arch: Arch = Arch.LARGE,
        with_register: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=f"dinov2_{backbone_arch}{'_reg' if with_register else ''}",
        )
        self.backbone.eval()
        self.backbone.cpu()
        self.patch_size: int = self.backbone.patch_size
        self.num_register_tokens: int = self.backbone.num_register_tokens
        self.image_shape = image_shape
        self.grid_shape = (
            self.image_shape[0] // self.patch_size,
            self.image_shape[1] // self.patch_size,
        )
        self.input_shape = (
            self.grid_shape[0] * self.patch_size,
            self.grid_shape[1] * self.patch_size,
        )
        self.preprocessor = T.Compose(
            (
                # T.ToTensor(),  # This OP does not support batch input.
                T.CenterCrop(self.input_shape),
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            )
        )

    def preprocess(self, x: BHWCUI8) -> BCHWF32:
        return self.preprocessor(
            torch.from_numpy(np.ascontiguousarray(x[..., ::-1]))
            .permute(0, 3, 1, 2)
            .float()
            / 255
        )

    @staticmethod
    def reverse_preprocessed(x: BCHWF32) -> BHWCUI8:
        return np.rint(
            (np.moveaxis(x.numpy(), 1, 3) * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN)
            * 255
        ).astype(np.uint8)[..., ::-1]

    @staticmethod
    def ensemble_augment(x: BHWCUI8) -> BHWCUI8:
        h_flip = x[:, :, ::-1]
        v_flip = x[:, ::-1, :]
        d_flip = v_flip[:, :, ::-1]
        return np.concatenate((x, h_flip, v_flip, d_flip), axis=0)

    def aggregate_ensembled_patchtokens(self, x: BTOKEN) -> BTOKEN:
        x, h_flip, v_flip, d_flip = x.reshape(4, *self.grid_shape, -1)
        return np.mean(
            a=(x, h_flip[:, ::-1], v_flip[::-1], d_flip[::-1, ::-1]),
            axis=0,
            keepdims=True,
        ).reshape(1, self.grid_shape[0] * self.grid_shape[1], -1)

    def aggregate_ensembled(self, x: BTOKEN) -> BTOKEN:
        """Counter part of `ensemble_augment()`."""
        n_orig, r = divmod(len(x), 4)
        assert r == 0
        return np.concatenate(
            [
                np.concatenate(
                    (
                        x[n::n_orig, 0:1].mean(axis=0, keepdims=True),
                        x[n::n_orig, 1 : self.num_register_tokens + 1].mean(
                            axis=0, keepdims=True
                        ),
                        self.aggregate_ensembled_patchtokens(
                            x[n::n_orig, self.num_register_tokens + 1 :]
                        ),
                    ),
                    axis=1,
                )
                for n in range(n_orig)
            ],
            axis=0,
        )

    def visualize_patchtokens(self, x: BPTOKEN) -> list[BGRUI8]:
        return [
            np.rint(
                (patch_token - (pmin := patch_token.min()))
                / (patch_token.max() - pmin)
                * 255
            )
            .astype(np.uint8)
            .reshape(*self.grid_shape, 3)
            for patch_token in x[..., :3]  # Take the first 3 channels to visualize.
        ]

    @torch.inference_mode()
    def forward(self, x: BCHWF32) -> Tensor:
        # ret = self.backbone.forward_features(x, masks=None)
        x = self.backbone.prepare_tokens_with_masks(x, masks=None)
        for blk in self.backbone.blocks:
            x = blk(x)
        x_norm = self.backbone.norm(x)
        return x_norm

    def inference(
        self,
        x: BHWCUI8,
        ensemble: bool = False,
    ) -> dict[str, BCTOKEN | BRTOKEN | BPTOKEN]:
        assert x.shape[1:3] == self.image_shape
        if ensemble:
            x = self.ensemble_augment(x)
        x_norm: BTOKEN = self.forward(self.preprocess(x)).numpy()
        if ensemble:
            x_norm = self.aggregate_ensembled(x_norm)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
        }


if __name__ == "__main__":
    from pathlib import Path
    import cv2

    def print_diff(x, y, title: str):
        print(f"{title}: {np.abs(x - y).mean()}")

    model = NaiveDino(image_shape=(480, 640))
    IMG_DIR = Path("/Users/yusong/Documents/venture/datasets/trichotrack/raw")

    img_bgr = cv2.imread(str(IMG_DIR.joinpath("2.jpeg")))
    img_bgr_en = model.ensemble_augment(np.expand_dims(img_bgr, axis=0))

    input_en = model.preprocess(img_bgr_en)
    rev_en = model.reverse_preprocessed(input_en)
    for n, (x, y) in enumerate(zip(img_bgr_en, rev_en)):
        print_diff(x[2:-2, 5:-5], y, f"rev-{n}")
        cv2.imshow(f"orig-{n}", x)
        cv2.imshow(f"rev-{n}", y)
    cv2.waitKey()

    c1 = model.inference(np.expand_dims(img_bgr, axis=0), ensemble=False)
    c2 = model.inference(img_bgr_en, ensemble=False)
    c3 = model.inference(np.expand_dims(img_bgr, axis=0), ensemble=True)
    for n, (x, y) in enumerate(
        zip(img_bgr_en, model.visualize_patchtokens(c2["x_norm_patchtokens"]))
    ):
        for key in ("x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens"):
            print_diff(c1[key][0], c2[key][n], f"{key}-{n}")
        cv2.imshow(f"input-{n}", x)
        cv2.imshow(f"token-{n}", y)
    for key in ("x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens"):
        print_diff(c1[key], c3[key], f"{key}-ensemble")
    cv2.imshow("token-ensemble", model.visualize_patchtokens(c3["x_norm_patchtokens"])[0])
    cv2.waitKey()
