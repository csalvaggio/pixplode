import argparse
import time
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# Module defaults (Modify to fit your most common use case)
EVENT_TYPE = "hover"
MINIMUM_SIZE_PIXEL = 4
SHAPE = "circle"
RESIZE_MAXIMUM_DIMENSION_TO = 256
OUTPUT_IMAGE_PATH = None


"""
pixplode - An Interactive Quadtree Renderer

Allows dynamic, interactive subdivision of image regions using a quadtree.
Supports click or hover events to trigger splits and visualizes the process.

Author:
    Carl Salvaggio

Copyright:
    Copyright (C) 2025, Rochester Institute of Technology
"""


class QuadNode:
    """
    Represents a node in a quadtree spatial partitioning structure.

    Each QuadNode defines a rectangular region with a (x, y) top-left 
    position, width, height, and optional depth level. It can be 
    subdivided into four child nodes, forming a hierarchical quadtree.

    Attributes:
        x (float): X-coordinate of the node's top-left corner.
        y (float): Y-coordinate of the node's top-left corner.
        width (float): Width of the node region.
        height (float): Height of the node region.
        depth (int): Depth level of the node in the quadtree hierarchy.
        children (List[QuadNode]): List of child QuadNode instances.
        patch (optional): Visual or graphical patch reference.
        color (optional): Color or data attribute associated with the node.

    Author:
        Carl Salvaggio

    Copyright:
        Copyright (C) 2025, Rochester Institute of Technology
    """

    def __init__(self,                     
                 x: float,      
                 y: float,              
                 width: float,
                 height: float,
                 depth: int = 0):
        """
        Initializes a QuadNode

        Args:
            x (float): X-coordinate of the node's top-left corner.
            y (float): Y-coordinate of the node's top-left corner.
            width (float): Width of the node region.
            height (float): Height of the node region.
            depth (int, optional): Depth level in the quadtree. Defaults to 0.
        """
        self.x = x                        
        self.y = y                        
        self.width = width
        self.height = height
        self.depth = depth                      
        self.children: List['QuadNode'] = []
        self.patch: Optional[patches.Patch] = None
        self.color: Optional[np.ndarray] = None

    def subdivide(self) -> None:
        """
        Subdivides the current node into four child QuadNode instances

        This creates four equally sized quadrants: northwest, northeast, 
        southwest, and southeast. If the node is already subdivided, 
        the method does nothing.
        """
        if self.children:
            return          
        hw, hh = self.width / 2, self.height / 2
        self.children = [                                                     
            QuadNode(self.x, self.y, hw, hh, self.depth + 1),
            QuadNode(self.x + hw, self.y, hw, hh, self.depth + 1),           
            QuadNode(self.x, self.y + hh, hw, hh, self.depth + 1),        
            QuadNode(self.x + hw, self.y + hh, hw, hh, self.depth + 1),
        ]                  

    def is_leaf(self) -> bool:
        """
        Checks if the node is a leaf node
        """
        return not self.children


class InteractiveQuadtreeRenderer:
    """
    Interactive renderer for quadtree-based image subdivision

    Allows dynamic subdivision of image regions using a quadtree, with
    interactive click or hover events to trigger splits. Visualizes the
    process using matplotlib.

    Attributes:
        COOLDOWN (float): Minimum time (seconds) between hover-based
            subdivisions.
        image (np.ndarray): Input image array.
        shape (str): Shape type for visual patches ("rectangle" or
            "circle").
        minimum_size_pixel (int): Minimum pixel size for subdivision.
        event_type (str): Type of event to trigger subdivision ("click"
            or "hover").
        last_subdivide_time (float): Timestamp of last subdivision event.
        root (QuadNode): Root node of the quadtree.
        fig (matplotlib.figure.Figure): Matplotlib figure object.
        ax (matplotlib.axes.Axes): Matplotlib axes object.

    Author:
        Carl Salvaggio

    Copyright:
        Copyright (C) 2025, Rochester Institute of Technology
    """

    COOLDOWN = 0.05  # seconds

    def __init__(self,
                 image: np.ndarray,
                 shape: str,
                 minimum_size_pixel: int,
                 event_type: str,
                 output_image_path: str):
        """
        Initializes the interactive quadtree renderer

        Args:
            image (np.ndarray): Input image.
            shape (str): Shape type for patches ("rectangle" or "circle").
            minimum_size_pixel (int): Minimum pixel size for subdivision.
            event_type (str): Subdivision trigger type ("click" or
                "hover").
            output_image_path (str): Output image path.
        """
        self.image = image
        self.shape = shape
        self.minimum_size_pixel = minimum_size_pixel
        self.event_type = event_type
        self.output_image_path: Optional[str] = output_image_path
        self.last_subdivide_time = 0
        self.root = QuadNode(0, 0, image.shape[1], image.shape[0])
        self.fig, self.ax = self._setup_display()
        self._draw_initial_node(self.root)
        self._setup_events()

    def _compute_mean_color(self,
                            x1: int,
                            y1: int,
                            x2: int,
                            y2: int) -> np.ndarray:
        """
        Computes the mean color of a rectangular image region

        Args:
            x1 (int): Left boundary.
            y1 (int): Top boundary.
            x2 (int): Right boundary.
            y2 (int): Bottom boundary.

        Returns:
            np.ndarray: Mean RGB color as a NumPy array.
        """
        x2 = min(x2, self.image.shape[1])
        y2 = min(y2, self.image.shape[0])
        if x1 >= x2 or y1 >= y2:
            return np.zeros(3)
        region = self.image[y1:y2, x1:x2]
        if region.size == 0:
            return np.zeros(3)
        mean = np.mean(region, axis=(0, 1))
        mean = mean[::-1]  # Convert BGR (OpenCV) to RGB (matplotlib)
        return mean

    def _create_patch(self,
                      node: QuadNode) -> None:
        """
        Creates and adds a visual patch for a quadtree node

        Args:
            node (QuadNode): The quadtree node to visualize.
        """
        if self.shape == "rectangle":
            patch = patches.Rectangle(
                (node.x, node.y), node.width, node.height,
                linewidth=1, edgecolor=node.color, facecolor=node.color
            )
        elif self.shape == "circle":
            radius = min(node.width, node.height) / 2
            center_x = node.x + node.width / 2
            center_y = node.y + node.height / 2
            patch = patches.Circle(
                (center_x, center_y), radius,
                linewidth=1, edgecolor=node.color, facecolor=node.color
            )
        else:
            raise ValueError(f"Unknown shape: {self.shape}")

        self.ax.add_patch(patch)
        node.patch = patch

    def _node_bounds(self,
                     node: QuadNode) -> tuple[int, int, int, int]:
        """
        Calculates integer boundaries of a quadtree node

        Args:
            node (QuadNode): The node whose bounds to compute.

        Returns:
            tuple[int, int, int, int]: (x1, y1, x2, y2) boundaries.
        """
        x1, y1 = int(round(node.x)), int(round(node.y))
        x2 = min(int(round(node.x + node.width)), self.image.shape[1])
        y2 = min(int(round(node.y + node.height)), self.image.shape[0])
        return x1, y1, x2, y2

    def _draw_initial_node(self,
                           node: QuadNode) -> None:
        """
        Draws the initial patch for the root node

        Args:
            node (QuadNode): The root quadtree node.
        """
        x1, y1, x2, y2 = self._node_bounds(node)
        node.color = self._compute_mean_color(x1, y1, x2, y2)
        self._create_patch(node)

    def _find_leaf(self,
                   node: QuadNode,
                   x: float,
                   y: float) -> QuadNode:
        """
        Finds the leaf node containing the given point

        Args:
            node (QuadNode): Starting node.
            x (float): X-coordinate.
            y (float): Y-coordinate.

        Returns:
            QuadNode: The leaf node containing (x, y).
        """
        while not node.is_leaf():
            hw, hh = node.width / 2, node.height / 2
            if x < node.x + hw:
                node = node.children[0] if y < node.y + hh else \
                       node.children[2]
            else:
                node = node.children[1] if y < node.y + hh else \
                       node.children[3]
        return node

    def _subdivide_and_draw(self,
                            node: QuadNode) -> None:
        """
        Subdivides a node and draws its children

        Args:
            node (QuadNode): The node to subdivide.
        """
        if node.patch:
            node.patch.remove()
            node.patch = None
        node.subdivide()
        for child in node.children:
            x1, y1, x2, y2 = self._node_bounds(child)
            child.color = self._compute_mean_color(x1, y1, x2, y2)
            self._create_patch(child)
        self.fig.canvas.draw_idle()

    def _on_click(self,
                  event) -> None:
        """
        Handles mouse click events
        """
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        leaf = self._find_leaf(self.root, x, y)
        if (leaf.is_leaf() and
            min(leaf.width, leaf.height) >= 2 * self.minimum_size_pixel):
            self._subdivide_and_draw(leaf)

    def _on_hover(self,
                  event) -> None:
        """
        Handles mouse hover events
        """
        if (event.inaxes != self.ax or
            (time.time() - self.last_subdivide_time < self.COOLDOWN)):
            return
        x, y = event.xdata, event.ydata
        leaf = self._find_leaf(self.root, x, y)
        if (leaf.is_leaf() and
            min(leaf.width, leaf.height) >= 2 * self.minimum_size_pixel):
            self._subdivide_and_draw(leaf)
            self.last_subdivide_time = time.time()

    def _on_key_press(self, event):
        """
        Handles key press events
        """
        if event.key in ['escape', 'q', 'Q']:
            self._save()
            plt.close(self.fig)

    def _setup_display(self):
        """
        Sets up the matplotlib figure and axes

        Returns:
            tuple: (figure, axes) for plotting.
        """
        height, width, _ = self.image.shape
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.12)
        padding_x = width * 0.01
        padding_y = height * 0.01
        ax.set_xlim(-padding_x, width + padding_x)
        ax.set_ylim(-padding_y, height + padding_y)
        ax.set_aspect("equal")
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.invert_yaxis()
        ax.axis("off")
        help_text = "Press ESC, q, or Q to quit"
        self.help_text_artist = (
            fig.text(0.5, 0.05, help_text,
                     ha='center', va='bottom',
                     fontsize=10,
                     bbox=dict(boxstyle='round',
                               facecolor='white', alpha=0.7))
        )
        return fig, ax

    def _setup_events(self):
        """
        Sets up matplotlib event handlers
        """
        if self.event_type == "hover":
            self.fig.canvas.mpl_connect("motion_notify_event",
                                        self._on_hover)
        else:
            self.fig.canvas.mpl_connect("button_press_event",
                                        self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",
                                    self._on_key_press)

    def _save(self) -> None:
        """
        Save current image rendering to provided output image path
        """
        if self.help_text_artist:
            self.help_text_artist.remove()
        if self.output_image_path:
            self.fig.savefig(self.output_image_path,
                             dpi=300, bbox_inches='tight')
            print(f"Current rendering saved to: {self.output_image_path}")

    def show(self) -> None:
        """
        Displays the interactive plot
        """
        plt.show()


def main() -> None:
    """
    Entry point for the interactive quadtree renderer

    Parses command-line arguments, loads and resizes the input image,
    and launches the interactive rendering interface.

    Raises:
        SystemExit: If the image file does not exist or fails to load.
    """
    description = (
        "Interactively render an image to finer and finer resolution"
    )
    parser = argparse.ArgumentParser(description=description)

    help_message = "Path to the image file to render"
    parser.add_argument(
        "image_path",
        help=help_message
    )

    help_message = (
        f"Event type that triggers a split [default is {EVENT_TYPE}]"
    )
    parser.add_argument(
        "-e", "--event-type",
        choices=["hover", "click"],
        default=EVENT_TYPE,
        help=help_message
    )

    help_message = (
        f"Minimum size image element to display "
        f"[default is {MINIMUM_SIZE_PIXEL}]"
    )
    parser.add_argument(
        "-m", "--minimum-size-pixel",
        type=int,
        default=MINIMUM_SIZE_PIXEL,
        help=help_message
    )

    help_message = (
        f"Shape of the image element to display [default is {SHAPE}]"
    )
    parser.add_argument(
        "-s", "--shape",
        choices=["circle", "rectangle"],
        default=SHAPE,
        help=help_message
    )

    help_message = (
        f"Desired size for the maximum image dimension "
        f"[default is {RESIZE_MAXIMUM_DIMENSION_TO}]"
    )
    parser.add_argument(
        "-r", "--resize-maximum-dimension-to",
        type=int,
        default=RESIZE_MAXIMUM_DIMENSION_TO,
        help=help_message
    )

    help_message = (
        f"Output file path to save image to on exit "
        f"[default is {OUTPUT_IMAGE_PATH}]"
    )
    parser.add_argument(
        "-o", "--output-image-path",
        type=str,
        default=OUTPUT_IMAGE_PATH,
        help=help_message
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        parser.error(f"File does not exist: {args.image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Failed to read image: {args.image_path}")

    image = image.astype(float) / 255
    original_height, original_width = image.shape[:2]
    if args.resize_maximum_dimension_to > 0:
        current_max_dim = max(original_height, original_width)
        if current_max_dim > args.resize_maximum_dimension_to:
            scale = args.resize_maximum_dimension_to / current_max_dim
            image = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

    renderer = InteractiveQuadtreeRenderer(
        image=image,
        shape=args.shape,
        minimum_size_pixel=args.minimum_size_pixel,
        event_type=args.event_type,
        output_image_path=args.output_image_path
    )
    try:
        renderer.show()
    except KeyboardInterrupt:
        plt.close(renderer.fig)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
