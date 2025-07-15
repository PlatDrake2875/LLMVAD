import matplotlib.pyplot as plt
import os
import logging


def plot_anomaly_scores(
    scores: list[float],
    video_filename: str = "anomaly_scores",
    output_dir: str = ".",
    dpi: int = 300,
):
    """
    Generates and saves a time-series-like plot of anomaly scores.
    """
    if not scores:
        logging.warning("No scores provided for plotting. Skipping graph generation.")
        return

    logging.info(f"Generating anomaly score plot for {video_filename}...")

    x_values = range(len(scores))

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, scores, marker="o", linestyle="-", color="blue", markersize=4)

    plt.title(f"Anomaly Scores Over Time for {video_filename}", fontsize=16)
    plt.xlabel("Chunk Index", fontsize=12)
    plt.ylabel("Anomaly Score (0-1)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(-0.05, 1.05)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    anomaly_threshold = 0.7
    for i, score in enumerate(scores):
        if score > anomaly_threshold:
            plt.axvline(x=i, color="red", linestyle=":", linewidth=1, alpha=0.5)
            plt.text(
                i,
                score + 0.05,
                f"{score:.2f}",
                color="red",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plot_filename = f"{os.path.splitext(video_filename)[0]}_anomaly_scores.png"
    save_path = os.path.join(output_dir, plot_filename)

    try:
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        logging.info(f"Anomaly score plot saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error saving anomaly score plot to {save_path}: {e}")
    finally:
        plt.close()
