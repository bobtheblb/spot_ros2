import sys

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from synchros2.utilities import namespace_with


class GlobalGridStitcher(Node):
    def __init__(self, robot_name: str = None):
        super().__init__("global_grid_stitcher")
        self.robot_name = robot_name

        # Parameters
        self.global_grid_resolution = 0.5  # meters per cell
        self.global_grid_size = 1500        # 1000x1000 cells
        self.height_scale = 20.0            # decode: height_m = cell / height_scale
        self.center_x = self.global_grid_size // 2
        self.center_y = self.global_grid_size // 2

        # Initialize empty global grid with NaN-like sentinel (-128 = unknown)
        self.global_grid = np.full(
            (self.global_grid_size, self.global_grid_size), -128, dtype=np.int8
        )

        # Publisher
        self.global_pub = self.create_publisher(
            OccupancyGrid, namespace_with(robot_name, "global_grid"), 1
        )

        # Subscriber
        self.local_sub = self.create_subscription(
            OccupancyGrid,
            namespace_with(robot_name, "terrain"),
            self.local_grid_callback,
            1,
        )

    def local_grid_callback(self, msg: OccupancyGrid):
        """Stitch local grid into global heightmap using grid_msg.info.origin."""
        print("Running local grid callback", flush=True)

        height = msg.info.height
        width = msg.info.width
        resolution = msg.info.resolution

        # Decode int8 back to height in meters, keeping -128 as unknown
        raw = np.array(msg.data, dtype=np.int8).reshape((height, width))
        unknown_mask = raw == -128

        # Origin of local grid in vision frame (position of cell (0,0) corner)
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Cell centers in vision frame
        # col indices → x, row indices → y
        col_indices = np.arange(width)
        row_indices = np.arange(height)

        xs = origin_x + (col_indices + 0.5) * resolution  # shape (width,)
        ys = origin_y + (row_indices + 0.5) * resolution  # shape (height,)

        # Convert vision-frame positions to global grid indices
        gx = np.round(xs / self.global_grid_resolution).astype(int) + self.center_x  # (width,)
        gy = np.round(ys / self.global_grid_resolution).astype(int) + self.center_y  # (height,)

        # Build full 2D index grids via broadcasting
        gx_grid = np.tile(gx[np.newaxis, :], (height, 1))  # (height, width)
        gy_grid = np.tile(gy[:, np.newaxis], (1, width))   # (height, width)

        # Mask out-of-bounds and unknown cells
        in_bounds = (
            (gx_grid >= 0) & (gx_grid < self.global_grid_size) &
            (gy_grid >= 0) & (gy_grid < self.global_grid_size)
        )
        valid = in_bounds & ~unknown_mask

        # Write valid cells into global grid
        self.global_grid[gy_grid[valid], gx_grid[valid]] = raw[valid]

        self.publish_global_grid(msg)

    def publish_global_grid(self, msg: OccupancyGrid):
        """Publish the stitched global heightmap as an OccupancyGrid."""
        global_msg = OccupancyGrid()
        global_msg.header.stamp = self.get_clock().now().to_msg()
        global_msg.header.frame_id = namespace_with(self.robot_name, "vision")

        global_msg.info = MapMetaData()
        global_msg.info.resolution = self.global_grid_resolution
        global_msg.info.width = self.global_grid_size
        global_msg.info.height = self.global_grid_size

        # Origin: center of global grid maps to (0, 0) in vision frame
        global_msg.info.origin.position.x = -self.center_x * self.global_grid_resolution
        global_msg.info.origin.position.y = -self.center_y * self.global_grid_resolution
        global_msg.info.origin.orientation.w = 1.0

        global_msg.data = self.global_grid.flatten().tolist()
        self.global_pub.publish(global_msg)


def main(args=None):
    if args is None:
        args = sys.argv
    rclpy.init(args=args)
    node = GlobalGridStitcher(robot_name=None)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)