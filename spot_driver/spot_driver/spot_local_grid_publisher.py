#!/usr/bin/env python3
import sys
from typing import List, Optional

import numpy as np
import rclpy
import traceback
from bosdyn.api.local_grid_pb2 import LocalGrid
from bosdyn.client import create_standard_sdk
from bosdyn.client.common import FutureWrapper
from bosdyn.client.frame_helpers import GROUND_PLANE_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.time_sync import TimedOutError as TimeSyncTimedOutError
from builtin_interfaces.msg import Time
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node

from spot_driver.manual_conversions import se3_pose_to_ros_pose
from spot_driver.ros_helpers import get_from_env_and_fall_back_to_param
from spot_wrapper.wrapper import robotToLocalTime

# VALID_GRIDS = ["terrain", "terrain_valid", "intensity", "no_step", "obstacle_distance"]
VALID_GRIDS = ["terrain"]

CODE_VERSION = "v2"

class LocalGridPublisher(Node):
    def __init__(self) -> None:
        super().__init__("local_grid_publisher")
        self.get_logger().warn("Local grid node started!!!")
        self.get_logger().debug("Initializing LocalGridPublisher Node...")

        read_only = ParameterDescriptor(read_only=True)
        self.declare_parameter("local_grid_name", "terrain", read_only)
        self.grid_name = self.get_parameter("local_grid_name").value

        # RViz message filters are very sensitive to timestamp/TF alignment. Spot local-grid acquisition_time
        # can be in a different time domain than the ROS clock used by TF broadcasters, so default to ROS time.

        # OccupancyGrid.data is int8[]. For terrain height (meters), we store round(height_m * scale)
        # clamped to [-128,127]. Default scale=10 => decimeters (~±12.7 m range); decode: height_m = cell / scale.
        self.declare_parameter("terrain_height_scale", 100.0)
        # self.terrain_height_scale = float(self.get_parameter("terrain_height_scale").value)
        self.terrain_height_scale = 20.0

        # Verify the requested grid name is an actual grid name
        if self.grid_name not in VALID_GRIDS:
            self.get_logger().error(f'Requested grid "{self.grid_name}" is not a valid local_grid type!')
            raise ValueError("Invalid local_grid name")

        # Get robot Credentials
        self.username: str = get_from_env_and_fall_back_to_param("BOSDYN_CLIENT_USERNAME", self, "username", "user")
        self.password: str = get_from_env_and_fall_back_to_param("BOSDYN_CLIENT_PASSWORD", self, "password", "password")
        self.ip: str = get_from_env_and_fall_back_to_param("SPOT_IP", self, "hostname", "10.0.0.3")

        if not self.ip or not self.username or not self.password:
            self.get_logger().error("Robot credentials not found")
            raise ValueError("Robot credentials not found")

        # Verify the credentials work
        self.get_logger().debug("🧰 Creating SDK objects...")
        self.sdk = create_standard_sdk("local_grid_publisher")
        self.robot = self.sdk.create_robot(self.ip)
        self.get_logger().debug("🧰 Created SDK objects successfully!")

        self.get_logger().debug("🔐 Attempting authentication...")
        self.robot.authenticate(self.username, self.password)  # an exception will be raised if authentication fails
        self.get_logger().debug("🔐 Robot authenticated successfully!")

        self.get_logger().debug("🕰 Starting time sync thread...")
        self.robot.time_sync.wait_for_sync()
        self.get_logger().debug("🕰 Time sync successful!")

        # Create LocalGridClient
        self.get_logger().debug("Creating LocalGridClient...")
        self.local_grid_client = self.robot.ensure_client(LocalGridClient.default_service_name)
        self.get_logger().debug("LocalGridClient created successfully!")

        # Create RobotStateClient
        self.get_logger().debug("Creating RobotStateClient...")
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.get_logger().debug("RobotStateClient created successfully!")

        # Create ROS2 publisher
        self.get_logger().debug("📡 Creating OccupancyGrid publisher...")
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, "grid_topic_REMAP_ME", 10)
        self.get_logger().debug("📡 OccupancyGrid publisher created successfully!")

        # Indicate successful initialization
        self.get_logger().debug("[✓] Spot Local Grid Publisher Node initialized")

        # Set runtime variables
        self.first_draw_done = False
        self.im = None
        self.fig = None
        self.ax = None

        self.fetch_next_grid_data()

    def acquisition_to_ros_time(self, acquisition):
        local = robotToLocalTime(acquisition, self.robot)
        return Time(sec=int(local.seconds), nanosec=int(local.nanos))

    def _cells_to_occupancy(self, raw_cells: np.ndarray) -> np.ndarray:
        """
        Convert Spot local-grid cell values to ROS OccupancyGrid-compatible int8 values.

        ROS OccupancyGrid semantics: -1 unknown, 0 free, 100 occupied.
        Spot local grids are not strictly occupancy grids, so we apply a best-effort mapping.

        Terrain height (meters from Spot after unpack_grid): stored as round(height_m * terrain_height_scale)
        clamped to int8 (default scale 10 => decimeters; decode height_m = value / terrain_height_scale).
        """
        if raw_cells is None:
            raise ValueError("Local grid cell data is empty")

        def terrain_height_to_int8(height_m: np.ndarray) -> np.ndarray:
            """Quantize terrain height in meters into int8 OccupancyGrid cells."""
            out = np.full(height_m.shape, -128, dtype=np.int8)
            finite = np.isfinite(height_m)
            if self.terrain_height_scale <= 0.0:
                raise ValueError("terrain_height_scale must be positive")
            scaled = np.rint(height_m * self.terrain_height_scale)
            clipped = np.clip(scaled, -128, 127)
            out[finite] = clipped[finite].astype(np.int8)
            return out

        # Handle floating-point grids (common when cell_value_scale/offset are applied).
        if np.issubdtype(raw_cells.dtype, np.floating):
            cells = raw_cells.astype(np.float32, copy=False)
            occ = np.full(cells.shape, -1, dtype=np.int8)

            finite = np.isfinite(cells)

            assert self.grid_name == "terrain", "Only supports terrain height at the moment"

            # The Spot SDK example treats obstacle_distance as meters where:
            # - inside obstacle: <= 0.0
            # - border band: (0.0, 0.33)
            # - free-ish: >= 0.33
            if self.grid_name == "obstacle_distance":
                occ[np.logical_and(finite, cells <= 0.0)] = 100
                occ[np.logical_and(finite, np.logical_and(cells > 0.0, cells < 0.33))] = 50
                occ[np.logical_and(finite, cells >= 0.33)] = 0
                return occ

            # The Spot SDK visualizer treats no_step as steppable if > 0.0.
            if self.grid_name == "no_step":
                occ[np.logical_and(finite, cells > 0.0)] = 0
                occ[np.logical_and(finite, cells <= 0.0)] = 100
                return occ

            # Terrain: actual height in meters (Spot SDK / basic_streaming_visualizer uses unpack as float height).
            if self.grid_name == "terrain":
                return terrain_height_to_int8(cells.astype(np.float64, copy=False))

            # Other float grids (unexpected): min-max visualization fallback only.
            finite_vals = cells[finite]
            if finite_vals.size == 0:
                return occ
            vmin = float(np.percentile(finite_vals, 1))
            vmax = float(np.percentile(finite_vals, 99))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                return occ
            scaled = (cells - vmin) / (vmax - vmin)
            scaled = np.clip(scaled, 0.0, 1.0)
            occ[finite] = (scaled[finite] * 100.0).astype(np.int8)
            return occ
        else:
            raise Exception("Height map must be of floating type")

        # Handle uint8 grids: keep relative ordering but shift into signed range.
        if raw_cells.dtype == np.uint8:
            return (raw_cells.astype(np.int16) - 128).astype(np.int8)

        # Handle signed integer grids: clamp to int8 range (avoid wrap-around).
        if np.issubdtype(raw_cells.dtype, np.integer):
            clipped = np.clip(raw_cells.astype(np.int16, copy=False), -128, 127)
            return clipped.astype(np.int8)

        raise TypeError(f"Unsupported local grid dtype: {raw_cells.dtype}")

    def fetch_next_grid_data(self) -> None:
        future = self.local_grid_client.get_local_grids_async([self.grid_name])
        future.add_done_callback(self.publish_grid)

    def publish_grid(self, future: FutureWrapper) -> None:
        """
        Converts the local grid protobuf into a ROS occupancy grid message

        Depending on the requested grid, some conversions may be done, as ROS OccupancyGrid data must be in int8 format

        Code in this function is adapted from the Boston Dynamics Spot SDK basic_streaming_visualizer example
        """
        proto = future.result()
        local_grid_proto = None
        for local_grid_found in proto:
            if local_grid_found.local_grid_type_name == self.grid_name:
                local_grid_proto = local_grid_found

        if local_grid_proto is None:
            self.get_logger().error(f'Did not receive local grid "{self.grid_name}" in response')
            self.fetch_next_grid_data()
            return

        # Populate Grid data and convert datatype if necessary

        raw_cells = self.unpack_grid(local_grid_proto)
        # self.get_logger().info(f"raw cells | min = {raw_cells.min()}, max = {raw_cells.max()}")

        self.get_logger().info(f"raw_cells | shape = {raw_cells.shape}, min = {raw_cells.min()}, max = {raw_cells.max()}")
        converted_cells = self._cells_to_occupancy(raw_cells)
        self.get_logger().info(f"converted_cells | shape = {converted_cells.shape}, min = {converted_cells.min()}, max = {converted_cells.max()}")

        # grid = converted_cells.reshape(
        #     local_grid_proto.local_grid.extent.num_cells_y, local_grid_proto.local_grid.extent.num_cells_x
        # )

        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = VISION_FRAME_NAME

        ros_time = self.acquisition_to_ros_time(local_grid_proto.local_grid.acquisition_time)
        grid_msg.header.stamp = ros_time
        grid_msg.info.map_load_time = ros_time

        # Spatial information
        grid_msg.info.resolution = local_grid_proto.local_grid.extent.cell_size
        grid_msg.info.width = int(local_grid_proto.local_grid.extent.num_cells_x)
        grid_msg.info.height = int(local_grid_proto.local_grid.extent.num_cells_y)

        transform = get_a_tform_b(
            local_grid_proto.local_grid.transforms_snapshot,
            VISION_FRAME_NAME,
            local_grid_proto.local_grid.frame_name_local_grid_data,
        )

        # self.get_logger().info(f"transform | x = {transform.position.x}, y = {transform.position.y}, z = {transform.position.z}")

        # Don't need this
        # OccupancyGrid's origin is the pose of cell (0,0) corner. Spot local grid is cell-centered, so offset by half-cell.
        # transform.x += 0.5 * grid_msg.info.resolution
        # transform.y += 0.5 * grid_msg.info.resolution

        grid_msg.info.origin = se3_pose_to_ros_pose(transform)
        grid_msg.data = converted_cells.astype(np.int8, copy=False).tolist()
        decoded_data = np.array(grid_msg.data) / self.terrain_height_scale
        self.get_logger().info(f"grid_msg.data | shape = {len(decoded_data)}, min = {np.min(decoded_data)}, max = {np.max(decoded_data)}")

        min_height = np.nanmin(grid_msg.data)
        max_height = np.nanmax(grid_msg.data)
        self.get_logger().info(f"min_height | min_height = {min_height}, max_height = {max_height}")

        # Publish and begin the next fetch
        self.occupancy_grid_pub.publish(grid_msg)
        self.fetch_next_grid_data()

    # Helper functions for local grid processing - functions taken from Bosdyn Dynamics Spot SDK visualizer example
    def unpack_grid(self, local_grid_proto: LocalGrid) -> np.array:
        """Unpack the local grid proto."""
        # Determine the data type for the bytes data.
        data_type = self.get_numpy_data_type(local_grid_proto.local_grid)
        if data_type is None:
            print("Cannot determine the dataformat for the local grid.")
            return None
        # Decode the local grid.
        if local_grid_proto.local_grid.encoding == LocalGrid.ENCODING_RAW:
            full_grid = np.frombuffer(local_grid_proto.local_grid.data, dtype=data_type)
        elif local_grid_proto.local_grid.encoding == LocalGrid.ENCODING_RLE:
            full_grid = self.expand_data_by_rle_count(local_grid_proto, data_type=data_type)
        else:
            # Return nothing if there is no encoding type set.
            return None
        # Apply the offset and scaling to the local grid.
        if local_grid_proto.local_grid.cell_value_scale == 0:
            return full_grid
        full_grid_float = full_grid.astype(np.float64)
        full_grid_float *= local_grid_proto.local_grid.cell_value_scale
        full_grid_float += local_grid_proto.local_grid.cell_value_offset
        return full_grid_float

    def get_numpy_data_type(self, local_grid_proto: LocalGrid) -> np.dtype:
        """Convert the cell format of the local grid proto to a numpy data type."""
        if local_grid_proto.cell_format == LocalGrid.CELL_FORMAT_UINT16:
            return np.uint16
        elif local_grid_proto.cell_format == LocalGrid.CELL_FORMAT_INT16:
            return np.int16
        elif local_grid_proto.cell_format == LocalGrid.CELL_FORMAT_UINT8:
            return np.uint8
        elif local_grid_proto.cell_format == LocalGrid.CELL_FORMAT_INT8:
            return np.int8
        elif local_grid_proto.cell_format == LocalGrid.CELL_FORMAT_FLOAT64:
            return np.float64
        elif local_grid_proto.cell_format == LocalGrid.CELL_FORMAT_FLOAT32:
            return np.float32
        else:
            return None

    def expand_data_by_rle_count(self, local_grid_proto: LocalGrid, data_type: np.dtype = np.int16) -> np.array:
        """Expand local grid data to full bytes data using the RLE count."""
        cells_pz = np.frombuffer(local_grid_proto.local_grid.data, dtype=data_type)
        cells_pz_full = []
        # For each value of rle_counts, we expand the cell data at the matching index
        # to have that many repeated, consecutive values.
        for i in range(0, len(local_grid_proto.local_grid.rle_counts)):
            for j in range(0, local_grid_proto.local_grid.rle_counts[i]):
                cells_pz_full.append(cells_pz[i])
        return np.array(cells_pz_full)

def main(args: Optional[List[str]] = None) -> None:
    if args is None:
        args = sys.argv
    rclpy.init(args=args)
    try:
        node = LocalGridPublisher()
    except Exception:
        try:
            rclpy.logging.get_logger("local_grid_publisher").error(traceback.format_exc())
        except Exception:
            pass
        rclpy.shutdown()
        return

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
