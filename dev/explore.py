from pathlib import Path

import cv2
import numpy as np


IMG_DIR = Path("/Users/yusong/Documents/venture/datasets/trichotrack/raw")
TOKEN_PATH = Path("/Users/yusong/Documents/dinov2/dev/vitl14-ensemble/condensed_0_9999.npz")


tokens = np.load(TOKEN_PATH)
cls_tokens = tokens["cls_tokens"]
pat_tokens = tokens["pat_tokens"]
cls_tokens /= np.linalg.norm(cls_tokens, axis=1, keepdims=True)
pat_tokens /= np.linalg.norm(pat_tokens, axis=1, keepdims=True)


def query(id, by_cls: bool = True, k: int = 5) -> list[tuple[int, float]]:
    tokens = cls_tokens if by_cls else pat_tokens
    pairwise_similarities = tokens.dot(tokens[id : id + 1].T).squeeze()
    similarity_spectrum = np.argsort(pairwise_similarities)
    nearest_ids = similarity_spectrum[-k - 1 : -1]
    farthest_ids = similarity_spectrum[:k]
    results = [(i, pairwise_similarities[i]) for i in nearest_ids[::-1]]
    results.extend((i, pairwise_similarities[i]) for i in farthest_ids[::-1])
    return results


def visualize_query_result(
    result: list[tuple[int, float]],
    include_opposite: bool = True,
) -> np.ndarray:
    if include_opposite:
        n_rows, n_cols = 2, len(result) // 2
    else:
        n_rows, n_cols = 1, len(result) // 2
        result = result[:n_cols]
    images = [
        cv2.putText(
            img=cv2.imread(str(IMG_DIR.joinpath(f"{img_id}.jpeg"))),
            text=f"{img_id}: {similarly:.3f}",
            org=(20, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0) if n < n_cols else (0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        for n, (img_id, similarly) in enumerate(result)
    ]
    if include_opposite:
        return np.vstack((np.hstack(images[:n_cols]), np.hstack(images[n_cols:])))
    else:
        return np.hstack(images[:n_cols])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script to explore DINOv2 features.",
        add_help=True,
    )
    parser.add_argument(
        "-q",
        "--query",
        required=True,
        type=int,
        help="ID of the queried image.",
    )
    parser.add_argument(
        "-k",
        "--k_neighbour",
        required=False,
        default=5,
        type=int,
        help="Number of neighbours to be retrieved.",
    )
    parser.add_argument(
        "-o",
        "--include_opposite",
        required=False,
        action="store_true",
        help="Whether include opposite result in retrieved result.",
    )
    args = parser.parse_args()

    if (qid := args.query) < 0 or qid >= len(cls_tokens):
        print(f"Query (ID) shall be within (0, {len(cls_tokens)-1})")

    if (k := args.k_neighbour) < 1:
        print("K neighbour shall be positive.")

    query_img = cv2.imread(str(IMG_DIR.joinpath(f"{qid}.jpeg")))
    cls_img = visualize_query_result(
        result=query(
            id=qid,
            by_cls=True,
            k=k,
        ),
        include_opposite=args.include_opposite,
    )
    pat_img = visualize_query_result(
        result=query(
            id=qid,
            by_cls=False,
            k=k,
        ),
        include_opposite=args.include_opposite,
    )
    cv2.imshow("Query", query_img)
    cv2.imshow("Class", cls_img)
    cv2.imshow("Patch", pat_img)
    cv2.waitKey()
