import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

def save_plot(file_path: str, save_plots: bool, file_types=["png"], dpi: float = 200, transparent=False):
    """Saves the current plot to specified file formats.

    Parameters
    ----------
    file_path : str
        The base file path for the saved plot.
    save_plots : bool
        Whether to save the plot.
    file_types : list, optional
        A list of file extensions to save (e.g., ["png", "pdf"]). Default is ["png"].
    dpi : float, optional
        The dots per inch for image resolution. Defaults to 200.

    Raises
    ------
    ValueError
        If no plot exists to save.
    """

    if not plt.gcf().get_axes():
        raise ValueError("No plot to save.")

    if save_plots:
        if "png" in file_types:
            plt.savefig(f"{file_path}.png", dpi=dpi, bbox_inches="tight", transparent=transparent)
            print(f"plot was saved as: {file_path}.png")
        if "pdf" in file_types:
            plt.savefig(f"{file_path}.pdf", bbox_inches="tight", transparent=transparent)
            print(f"plot was saved as: {file_path}.pdf")
        if "jpg" in file_types:
            plt.savefig(f"{file_path}.jpg", bbox_inches="tight", transparent=transparent, dpi=dpi)
            print(f"plot was saved as: {file_path}.jpg")

def valuearray(dict : dict) -> np.ndarray:
    return np.array(list(dict.values()))


def create_subfigure_label(ax, letter: str, width: float, height: float, fontsize: float = 10, correction_text: list = [0, 0], box: bool = True, box_alpha: float =0.8) -> None:
    """
    Creates a subfigure label with a white background box in the upper left corner on a matplotlib Axes object.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The matplotlib Axes object on which to add the label.
    letter : str
        The letter (e.g., 'A', 'B', 'C') to display as the label.
    width : float
        The relative width of the label box within the Axes, as a fraction of the Axes width.
    height : float
        The relative height of the label box within the Axes, as a fraction of the Axes height.
    correction_text : list[float], optional
        A two-element list specifying a horizontal and vertical correction offset (in fractions of `width` and `height`)
        applied to the text position within the label box. Defaults to [0, 0] (no correction).

    Returns
    -------
    None
    """
    # Calculate text position within the label box
    x_text = width/2 + correction_text[0]
    y_text = 1 - height/2  + correction_text[1]

    # Add the label text to the Axes
    ax.text(x_text, y_text, letter, transform=ax.transAxes, fontsize=fontsize, fontweight="bold", color="black", zorder=50, horizontalalignment='center',
        verticalalignment='center',)
    
    if box:
        # Calculate the position and size of the label box
        y_box = 1 - height
        
        # Create a transparent white rectangle patch for the label box
        ax.add_patch(mpatches.Rectangle(xy=[0, y_box], width=width, height=height,
                                    facecolor='white', edgecolor=None,
                                    transform=ax.transAxes, lw=2, zorder=18, alpha=box_alpha))
    
    # Ensuring axis spines are on top
    for spine in ax.spines.values():
        spine.set_zorder(100)

def get_colors_from_colormap(cmap, N, start=0, stop=1):
    """
    Gets a list of N colors from a given Matplotlib colormap.

    Args:
        cmap: The Matplotlib colormap to use.
        N: The number of colors to generate.

    Returns:
        A list of N color tuples (r, g, b, alpha).
    """

    cmap = plt.cm.get_cmap(cmap)
    colors = [cmap(i / (N - 1) * (stop - start) + start) for i in range(N)]
    return colors