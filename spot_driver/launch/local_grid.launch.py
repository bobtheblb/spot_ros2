import launch
import launch_ros
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

THIS_PACKAGE = "spot_driver"


def generate_launch_description() -> launch.LaunchDescription:
    # Define launch arguments
    local_grid_name = DeclareLaunchArgument(
        "local_grid_name", default_value="obstacle_distance", description="Name of the local_grid you want published"
    )

    spot_name = DeclareLaunchArgument("spot_name", default_value="", description="Name of Spot")

    stamp_with_ros_time = DeclareLaunchArgument(
        "stamp_with_ros_time",
        default_value="true",
        description="If true, stamp the grid with ROS time (better TF/Rviz alignment).",
    )
    terrain_height_scale = DeclareLaunchArgument(
        "terrain_height_scale",
        default_value="100.0",
        description="Terrain height encoding: int8 cell = round(height_m * scale), clamped. Default 10 = decimeters.",
    )
    time_sync_timeout_sec = DeclareLaunchArgument(
        "time_sync_timeout_sec",
        default_value="30.0",
        description="Seconds to wait for Spot SDK time sync (default SDK wait is only 3s).",
    )

    local_grid_topic = PathJoinSubstitution([LaunchConfiguration("spot_name"), LaunchConfiguration("local_grid_name")])

    local_grid_node = launch_ros.actions.Node(
        package="spot_driver",
        executable="spot_local_grid_publisher_node",
        output="screen",
        # Without a TTY, Python/rcutils often block-buffer stderr; logs then appear late or never in the launch terminal.
        emulate_tty=True,
        parameters=[
            {
                "local_grid_name": LaunchConfiguration("local_grid_name"),
                "stamp_with_ros_time": LaunchConfiguration("stamp_with_ros_time"),
                "terrain_height_scale": LaunchConfiguration("terrain_height_scale"),
                "time_sync_timeout_sec": LaunchConfiguration("time_sync_timeout_sec"),
            }
        ],
        namespace=LaunchConfiguration("spot_name"),
        remappings=[("grid_topic_REMAP_ME", local_grid_topic)],
    )

    return launch.LaunchDescription(
        [
            local_grid_name,
            spot_name,
            stamp_with_ros_time,
            terrain_height_scale,
            time_sync_timeout_sec,
            local_grid_node,
        ]
    )
