import matplotlib.pyplot as plt
import os
import logging

def plot_anomaly_scores(scores: list[float], video_filename: str = "anomaly_scores", output_dir: str = ".", dpi: int = 300):
    """
    Generates and saves a time-series-like plot of anomaly scores.

    Args:
        scores (list[float]): A list of anomaly scores.
        video_filename (str): The base name of the video file, used for plot title and output filename.
        output_dir (str): The directory where the plot image will be saved.
        dpi (int): The resolution of the saved image.
    """
    if not scores:
        logging.warning("No scores provided for plotting. Skipping graph generation.")
        return

    logging.info(f"Generating anomaly score plot for {video_filename}...")

    # Create x-axis values (chunk indices)
    x_values = range(len(scores))

    plt.figure(figsize=(12, 6)) # Adjust figure size for better readability
    plt.plot(x_values, scores, marker='o', linestyle='-', color='blue', markersize=4)

    plt.title(f'Anomaly Scores Over Time for {video_filename}', fontsize=16)
    plt.xlabel('Chunk Index', fontsize=12)
    plt.ylabel('Anomaly Score (0-1)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.05, 1.05) # Set y-axis limits slightly beyond 0-1 for clarity
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Highlight potential anomalies (e.g., scores above a certain threshold)
    # You might want to make this threshold configurable
    anomaly_threshold = 0.7 
    for i, score in enumerate(scores):
        if score > anomaly_threshold:
            plt.axvline(x=i, color='red', linestyle=':', linewidth=1, alpha=0.5)
            plt.text(i, score + 0.05, f'{score:.2f}', color='red', ha='center', va='bottom', fontsize=8)

    # Save the plot
    plot_filename = f"{os.path.splitext(video_filename)[0]}_anomaly_scores.png"
    save_path = os.path.join(output_dir, plot_filename)
    
    try:
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.savefig(save_path, dpi=dpi)
        logging.info(f"Anomaly score plot saved to: {save_path}")
    except Exception as e:
        logging.error(f"Error saving anomaly score plot to {save_path}: {e}")
    finally:
        plt.close() # Close the plot to free up memory


