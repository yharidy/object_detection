import cv2
import numpy as np
import pandas as pd

from src.data.waymo.enums import (
    WAYMO_TO_DOMAIN_CAMERA_MAP,
    WAYMO_TO_DOMAIN_CLASS_MAP,
    WAYMO_TO_DOMAIN_LIDAR_MAP,
    ClassID,
    WaymoCamera,
    WaymoLidar,
)
from src.data.waymo.lidar_transforms import range_image_to_point_cloud
from src.domain import (
    Box2D,
    Box3D,
    CameraCalibration,
    CameraImage,
    CameraIntrinsicsBrownConrady,
    CameraIntrinsicsPinhole,
    CameraLabel,
    CameraPosition,
    Frame,
    LidarCalibration,
    LidarLabel,
    LidarPointCloud,
    LidarPosition,
    LidarRangeImage,
    SensorRig,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class WaymoFrameParser:
    """Parser for Waymo Open Dataset frames. Provides methods to parse camera images, lidar range images, and sensor calibrations from DataFrames loaded from the dataset."""

    def parse_frame(
        self,
        timestamp_micros: int,
        camera_images_df: pd.DataFrame | None,
        lidar_range_images_df: pd.DataFrame | None,
        lidar_calibrations: dict[WaymoLidar, LidarCalibration] | None = None,
        lidar_returns: list[int] | None = None,
        load_point_clouds: bool = False,
        camera_labels_df: pd.DataFrame | None = None,
        lidar_labels_df: pd.DataFrame | None = None,
    ) -> Frame:
        """Parse a frame from camera images and lidar range images DataFrames.

        Args:
            timestamp_micros: The timestamp of the frame in microseconds.
            camera_images_df: DataFrame containing camera image data.
            lidar_range_images_df: DataFrame containing lidar range image data.
            lidar_calibrations: Optional dictionary of lidar calibrations, required if load_point_clouds is True.
            lidar_returns: List of return counts to parse (e.g., [1, 2]). Defaults to [1].
            load_point_clouds: Whether to convert range images to point clouds. Defaults to False.
            camera_labels_df: Optional DataFrame containing camera label data.
            lidar_labels_df: Optional DataFrame containing lidar label data.

        Returns:
            A Frame object containing the parsed camera images and lidar range images.
        """
        logger.debug(f"Parsing frame at timestamp {timestamp_micros}")
        camera_images = (
            self.parse_camera_images(camera_images_df)
            if camera_images_df is not None
            else {}
        )
        lidar_range_images, lidar_point_clouds = self.parse_lidar_data(
            lidar_range_images_df,
            lidar_calibrations,
            lidar_returns,
            load_point_clouds,
        )
        camera_labels = (
            self.parse_camera_labels(camera_labels_df)
            if camera_labels_df is not None
            else {}
        )
        lidar_labels = (
            self.parse_lidar_labels(lidar_labels_df)
            if lidar_labels_df is not None
            else {}
        )
        return Frame(
            timestamp_micros=timestamp_micros,
            camera_images=camera_images,
            lidar_range_images=lidar_range_images,
            lidar_point_clouds=lidar_point_clouds,
            camera_labels=camera_labels,
            lidar_labels=lidar_labels,
        )

    def parse_sensor_rig(
        self,
        camera_calibration_df: pd.DataFrame | None = None,
        lidar_calibration_df: pd.DataFrame | None = None,
    ) -> SensorRig:
        """Parse sensor rig from camera and lidar calibration DataFrames.

        Args:
            camera_calibration_df: DataFrame containing camera calibration data. Optional.
            lidar_calibration_df: DataFrame containing lidar calibration data. Optional.

        Returns:
            A SensorRig object containing the parsed camera and lidar calibrations.

        Raises:
            ValueError: If both camera_calibration_df and lidar_calibration_df are None.
        """
        logger.debug("Parsing sensor rig")
        if camera_calibration_df is None and lidar_calibration_df is None:
            raise ValueError(
                "At least one of camera_calibration_df or lidar_calibration_df must be provided."
            )
        camera_calibrations = {}
        if camera_calibration_df is not None:
            for _, row in camera_calibration_df.iterrows():
                camera_name = row["key.camera_name"]
                intrinsic_matrix = self._parse_camera_intrinsics(row)
                extrinsic_matrix = self._parse_camera_extrinsics(row)
                camera_calibrations[
                    WAYMO_TO_DOMAIN_CAMERA_MAP[WaymoCamera(camera_name)]
                ] = CameraCalibration(
                    camera=WAYMO_TO_DOMAIN_CAMERA_MAP[WaymoCamera(camera_name)],
                    intrinsic_matrix=intrinsic_matrix,
                    extrinsic_matrix=extrinsic_matrix,
                )
        lidar_calibrations = {}
        if lidar_calibration_df is not None:
            for _, row in lidar_calibration_df.iterrows():
                lidar_name = row["key.laser_name"]
                extrinsic_matrix = self._parse_lidar_extrinsics(row)
                min_inclination, max_inclination = self._parse_lidar_inclination_angles(
                    row
                )
                beam_inclinations = self._parse_lidar_beam_inclinations(row)
                lidar_calibrations[
                    WAYMO_TO_DOMAIN_LIDAR_MAP[WaymoLidar(lidar_name)]
                ] = LidarCalibration(
                    lidar=WAYMO_TO_DOMAIN_LIDAR_MAP[WaymoLidar(lidar_name)],
                    extrinsic_matrix=extrinsic_matrix,
                    beam_inclination_min=min_inclination,
                    beam_inclination_max=max_inclination,
                    beam_inclinations=beam_inclinations,
                )

        return SensorRig(cameras=camera_calibrations, lidars=lidar_calibrations)

    def parse_camera_images(
        self, camera_images_df: pd.DataFrame
    ) -> dict[CameraPosition, CameraImage]:
        """Parse camera images from the DataFrame.

        Args:
            camera_images_df: DataFrame containing camera image data.

        Returns:
            A dictionary mapping CameraPosition enum to CameraImage objects.
        """
        logger.debug("Parsing camera images")
        camera_images = {}
        for _, row in camera_images_df.iterrows():
            camera_name: int = row["key.camera_name"]
            timestamp_micros: int = row["key.frame_timestamp_micros"]
            image_data: bytes = row["[CameraImageComponent].image"]  # binary JPEG data
            decoded_image: np.ndarray = self._decode_jpeg(image_data)
            camera_images[WAYMO_TO_DOMAIN_CAMERA_MAP[WaymoCamera(camera_name)]] = (
                CameraImage(
                    camera=WAYMO_TO_DOMAIN_CAMERA_MAP[WaymoCamera(camera_name)],
                    timestamp_micros=timestamp_micros,
                    image=decoded_image,
                )
            )
        return camera_images

    def _convert_range_image_to_nparray(
        self, range_image: list[float], shape: list[int]
    ) -> np.ndarray:
        """Convert flat range image list to numpy array.

        Waymo stores range images as a flat array of floats: [range, intensity, elongation, no_label_zone].

        Args:
            range_image: Flat list of float values.
            shape: Shape of the range image [H, W].

        Returns:
            Numpy array reshaped to (H, W, 4).
        """
        # Waymo stores range images as a flat array of floats: [range, intensity, elongation, no_label_zone]
        return np.asarray(range_image, dtype=np.float32).reshape(
            shape[0], shape[1], 4
        )  # reshape to (H, W, 4) where 4 corresponds to (range, intensity, elongation, no_label_zone)

    def parse_lidar_data(
        self,
        lidar_range_images_df: pd.DataFrame,
        lidar_calibrations: dict[LidarPosition, LidarCalibration] | None = None,
        lidar_returns: list[int] | None = None,
        convert_to_point_cloud: bool = False,
    ) -> tuple[
        dict[LidarPosition, list[LidarRangeImage]],
        dict[LidarPosition, list[LidarPointCloud]] | None,
    ]:
        """Parse lidar data from the DataFrame.

        Args:
            lidar_range_images_df: DataFrame containing lidar range image data.
            lidar_calibrations: Optional dictionary of lidar calibrations, required if convert_to_point_cloud is True.
            returns: List of return counts to parse (e.g., [1, 2]).
            convert_to_point_cloud: Whether to convert range images to point clouds. Defaults to False.

        Returns:
            A tuple containing:
                - A dictionary mapping Lidar enum to a list of LidarRangeImage objects.
                - An optional dictionary mapping Lidar enum to a list of LidarPointCloud objects, returned if convert_to_point_cloud is True.
        """
        logger.debug("Parsing lidar data")
        if convert_to_point_cloud and lidar_calibrations is None:
            raise ValueError(
                "Lidar calibrations must be provided when convert_to_point_cloud is True."
            )
        lidar_returns = lidar_returns or [1]  # default to return 1 if not specified
        frame_range_images = {}
        frame_point_clouds = {}
        for _, row in lidar_range_images_df.iterrows():
            lidar = WAYMO_TO_DOMAIN_LIDAR_MAP[WaymoLidar(row["key.laser_name"])]
            timestamp_micros: int = row["key.frame_timestamp_micros"]
            lidar_range_images = []
            lidar_point_clouds = []
            if 1 in lidar_returns:
                range_image_1 = self._convert_range_image_to_nparray(
                    row["[LiDARComponent].range_image_return1.values"],
                    row["[LiDARComponent].range_image_return1.shape"],
                )
                lidar_range_images.append(
                    LidarRangeImage(
                        lidar=lidar,
                        timestamp_micros=timestamp_micros,
                        range_image=range_image_1,
                        return_count=1,
                    )
                )
                if convert_to_point_cloud:
                    calib = lidar_calibrations[lidar]
                    point_cloud_1 = range_image_to_point_cloud(
                        range_image_1,
                        beam_inclinations=calib.beam_inclinations,
                        beam_inclination_min=calib.beam_inclination_min,
                        beam_inclination_max=calib.beam_inclination_max,
                    )
                    lidar_point_clouds.append(
                        LidarPointCloud(
                            lidar=lidar,
                            timestamp_micros=timestamp_micros,
                            point_cloud=point_cloud_1,
                            return_count=1,
                        )
                    )
            if 2 in lidar_returns:
                range_image_2 = self._convert_range_image_to_nparray(
                    row["[LiDARComponent].range_image_return2.values"],
                    row["[LiDARComponent].range_image_return2.shape"],
                )
                lidar_range_images.append(
                    LidarRangeImage(
                        lidar=lidar,
                        timestamp_micros=timestamp_micros,
                        range_image=range_image_2,
                        return_count=2,
                    )
                )
                if convert_to_point_cloud:
                    point_cloud_2 = range_image_to_point_cloud(
                        range_image_2,
                        beam_inclinations=calib.beam_inclinations,
                        beam_inclination_min=calib.beam_inclination_min,
                        beam_inclination_max=calib.beam_inclination_max,
                    )
                    lidar_point_clouds.append(
                        LidarPointCloud(
                            lidar=lidar,
                            timestamp_micros=timestamp_micros,
                            point_cloud=point_cloud_2,
                            return_count=2,
                        )
                    )
            if not lidar_range_images:
                logger.warning(
                    f"No range images found for lidar {lidar} at timestamp {timestamp_micros} for returns {lidar_returns}"
                )
            frame_range_images[lidar] = lidar_range_images
            frame_point_clouds[lidar] = lidar_point_clouds

        return frame_range_images, frame_point_clouds

    def _decode_jpeg(self, image_data: bytes) -> np.ndarray:
        """Decode JPEG image data to RGB numpy array.

        Args:
            image_data: Binary JPEG data.

        Returns:
            Decoded image as RGB numpy array.
        """
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)

    def _parse_camera_intrinsics(
        self, camera_calibration_row: pd.Series
    ) -> CameraIntrinsicsPinhole | CameraIntrinsicsBrownConrady:
        """Parse camera intrinsics from calibration row.

        Args:
            camera_calibration_row: Pandas Series containing calibration data.

        Returns:
            CameraIntrinsicsPinhole or CameraIntrinsicsBrownConrady based on available data.
        """
        if "[CameraCalibrationComponent].intrinsic.k1" in camera_calibration_row:
            return CameraIntrinsicsBrownConrady(
                width=camera_calibration_row["[CameraCalibrationComponent].width"],
                height=camera_calibration_row["[CameraCalibrationComponent].height"],
                focal_length_u=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.f_u"
                ],
                focal_length_v=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.f_v"
                ],
                principal_point_u=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.c_u"
                ],
                principal_point_v=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.c_v"
                ],
                radial_distortion_k1=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.k1"
                ],
                radial_distortion_k2=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.k2"
                ],
                radial_distortion_k3=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.k3"
                ],
                tangential_distortion_p1=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.p1"
                ],
                tangential_distortion_p2=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.p2"
                ],
            )
        else:
            return CameraIntrinsicsPinhole(
                width=camera_calibration_row["[CameraCalibrationComponent].width"],
                height=camera_calibration_row["[CameraCalibrationComponent].height"],
                focal_length_u=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.f_u"
                ],
                focal_length_v=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.f_v"
                ],
                principal_point_u=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.c_u"
                ],
                principal_point_v=camera_calibration_row[
                    "[CameraCalibrationComponent].intrinsic.c_v"
                ],
            )

    def _parse_extrinsic_matrix(self, values: list[float]) -> np.ndarray:
        """Parse extrinsic matrix from list of values.

        Args:
            values: List of 16 float values representing the 4x4 transformation matrix.

        Returns:
            4x4 numpy array.

        Raises:
            ValueError: If the list does not contain exactly 16 values.
        """
        if len(values) != 16:
            raise ValueError(f"Expected 16 values, got {len(values)}")
        return np.asarray(values, dtype=np.float64).reshape(4, 4)

    def _parse_camera_extrinsics(self, calibration_row: pd.Series) -> np.ndarray:
        """Parse camera extrinsic matrix from calibration row.

        Args:
            calibration_row: Pandas Series containing calibration data.

        Returns:
            4x4 extrinsic matrix as numpy array.
        """
        return self._parse_extrinsic_matrix(
            calibration_row["[CameraCalibrationComponent].extrinsic.transform"]
        )

    def _parse_lidar_extrinsics(self, calibration_row: pd.Series) -> np.ndarray:
        """Parse lidar extrinsic matrix from calibration row.

        Args:
            calibration_row: Pandas Series containing calibration data.

        Returns:
            4x4 extrinsic matrix as numpy array.
        """
        return self._parse_extrinsic_matrix(
            calibration_row["[LiDARCalibrationComponent].extrinsic.transform"]
        )

    def _parse_lidar_inclination_angles(
        self, calibration_row: pd.Series
    ) -> tuple[float, float]:
        """Parse lidar inclination angles from calibration row.

        Args:
            calibration_row: Pandas Series containing calibration data.

        Returns:
            Tuple of minimum and maximum inclination angles in radians.
        """
        return (
            calibration_row["[LiDARCalibrationComponent].beam_inclination.min"],
            calibration_row["[LiDARCalibrationComponent].beam_inclination.max"],
        )

    def _parse_lidar_beam_inclinations(self, calibration_row: pd.Series) -> np.ndarray:
        """Parse lidar beam inclination angles from calibration row.

        Args:
            calibration_row: Pandas Series containing calibration data.
        Returns:
            Numpy array of beam inclination angles in radians, shape (num_beams,).
        """
        return np.asarray(
            calibration_row["[LiDARCalibrationComponent].beam_inclination.values"],
            dtype=np.float32,
        )

    def parse_camera_labels(
        self, camera_labels_df: pd.DataFrame
    ) -> dict[CameraPosition, list[CameraLabel]]:
        """Parse camera labels from the DataFrame.

        Args:
            camera_labels_df: DataFrame containing camera label data.

        Returns:
            Dictionary mapping camera names to lists of camera labels.
        """
        camera_labels: dict[CameraPosition, list[CameraLabel]] = {}
        for _, row in camera_labels_df.iterrows():
            camera = WAYMO_TO_DOMAIN_CAMERA_MAP[WaymoCamera(row["key.camera_name"])]
            camera_labels.setdefault(camera, []).append(
                CameraLabel(
                    object_id=row["key.camera_object_id"],
                    object_class=WAYMO_TO_DOMAIN_CLASS_MAP[
                        ClassID(row["[CameraBoxComponent].type"])
                    ],
                    box_2d=Box2D(
                        center_x=row["[CameraBoxComponent].box.center.x"],
                        center_y=row["[CameraBoxComponent].box.center.y"],
                        width=row["[CameraBoxComponent].box.size.x"],
                        height=row["[CameraBoxComponent].box.size.y"],
                    ),
                )
            )
        return camera_labels

    def parse_lidar_labels(self, lidar_labels_df: pd.DataFrame) -> list[LidarLabel]:
        """Parse lidar labels from the DataFrame.

        Args:
            lidar_labels_df: DataFrame containing lidar label data.
        Returns:
            List of lidar labels.
        """
        lidar_labels: list[LidarLabel] = []
        for _, row in lidar_labels_df.iterrows():
            lidar_labels.append(
                LidarLabel(
                    object_id=row["key.laser_object_id"],
                    object_class=WAYMO_TO_DOMAIN_CLASS_MAP[
                        ClassID(row["[LiDARBoxComponent].type"])
                    ],
                    box_3d=Box3D(
                        center_x=row["[LiDARBoxComponent].box.center.x"],
                        center_y=row["[LiDARBoxComponent].box.center.y"],
                        center_z=row["[LiDARBoxComponent].box.center.z"],
                        width=row["[LiDARBoxComponent].box.size.x"],
                        length=row["[LiDARBoxComponent].box.size.y"],
                        height=row["[LiDARBoxComponent].box.size.z"],
                        heading=row["[LiDARBoxComponent].box.heading"],
                    ),
                    speed=np.array(
                        [
                            row["[LiDARBoxComponent].speed.x"],
                            row["[LiDARBoxComponent].speed.y"],
                            row["[LiDARBoxComponent].speed.z"],
                        ],
                        dtype=np.float32,
                    ),
                    acceleration=np.array(
                        [
                            row["[LiDARBoxComponent].acceleration.x"],
                            row["[LiDARBoxComponent].acceleration.y"],
                            row["[LiDARBoxComponent].acceleration.z"],
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        return lidar_labels
