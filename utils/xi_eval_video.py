import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from FeaturesExtractor import FeaturesExtractor


def compute_xi_score(
    bisenet: list[list[list[float]]],
    bisenet_semcc: list[list[list[float]]],
    ctcm: list[list[list[float]]]
) -> list[float]:
    """
    Compute Xi-score for three groups of metrics:
    - bisenet
    - bisenet_semcc
    - ctcm

    Formula is derived from normalized average metrics of each group.
    """
    def average(values: list[float]) -> float:
        return sum(values) / len(values)

    def normalize(components: list[float]) -> list[float]:
        total = sum(components)
        return [x / total for x in components]

    # Compute averages for each metric group
    averages = [
        [average(metric) for metric in group]
        for group in [bisenet, bisenet_semcc, ctcm]
    ]

    # Normalize across corresponding metric positions
    normalized = [normalize(avg) for avg in zip(*averages)]

    print("ro, rs, scv, gc")
    for n in zip(*normalized):
        print(n)

    # Compute Xi-score based on normalized metric groups
    scores = [(n[0] + n[1]) / (n[2] * n[3]) for n in zip(*normalized)]
    total_score = sum(scores)
    return [score / total_score for score in scores]


def violins_plot(data: list[np.ndarray], title: str, labels: list[str]) -> None:
    """
    Create violin plots for metric distributions across different segmentation methods.
    """
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame({
        "Data": np.concatenate(data),
        "Group": np.repeat(labels, [len(d) for d in data])
    })

    ax = sns.violinplot(
        x="Group", y="Data", data=df,
        palette="Set3", bw_adjust=0.2, inner="quartile"
    )

    # Plot mean and median for each group
    for i, d in enumerate(data):
        mean_val, median_val = np.mean(d), np.median(d)
        ax.scatter(i, mean_val, color="red", marker="o", label="Mean" if i == 0 else "")
        ax.scatter(i, median_val, color="blue", marker="D", label="Median" if i == 0 else "")

    plt.legend()
    plt.title(title)
    plt.show()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute Xi-score and visualization metrics for segmentation models."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file with binary segmentation masks denoting rails (e.g. data/validation_masks.mp4)"
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Enable training mode for GMM."
    )
    parser.add_argument(
        "--silhouette",
        action="store_true",
        help="Compute silhouette score."
    )
    parser.add_argument(
        "--violins",
        action="store_true",
        help="Display violin plots for metric distributions."
    )
    parser.add_argument(
        "--components",
        type=int,
        default=5,
        help="Number of GMM components (default: 5)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Initialize feature extractor
    fe = FeaturesExtractor(args.video_path, n_components=args.components, training=args.training)
    collected_images, collected_bisenet, collected_bisenet_semcc, collected_ctcm = fe.get_sample_images()

    # Train GMM if in training mode
    if fe.training:
        fe.train_gmm(collected_images)

    # Optionally compute silhouette score
    if args.silhouette:
        print(fe.n_components, fe.get_silhouette(collected_images))

    # Initialize metrics containers
    bisenet = [[] for _ in range(4)]
    bisenet_semcc = [[] for _ in range(4)]
    ctcm = [[] for _ in range(4)]

    # Compute all feature scores
    for i in tqdm(range(len(collected_images)), desc="Computing scores"):
        for target, data in zip(
            [bisenet, bisenet_semcc, ctcm],
            [collected_bisenet, collected_bisenet_semcc, collected_ctcm],
        ):
            scores = fe.get_score(collected_images[i], data[i])
            for j in range(4):
                target[j].append(scores[j])

    # Compute Xi-score
    print("Xi Score:", compute_xi_score(bisenet, bisenet_semcc, ctcm))

    # Optionally visualize violin plots
    if args.violins:
        metric_titles = [
            "Metrics Distribution (Rails-Background Distance)",
            "Metrics Distribution (Rails-Extended Distance)",
            "Metrics Distribution (Rails Colour Variance)",
            "Metrics Distribution (Rails Geometrical Complexity)",
        ]
        metric_labels = ["BiSeNetV2", "BiSeNetV2 SEM-CC", "CTCM"]

        for i, title in enumerate(metric_titles):
            violins_plot(
                [bisenet[i], bisenet_semcc[i], ctcm[i]],
                title,
                metric_labels,
            )



if __name__ == "__main__":
    main()
