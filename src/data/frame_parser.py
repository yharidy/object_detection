import cv2
import numpy as np
import pandas as pd
from enums import Camera, Lidar

from .models import (
    CameraCalibration,
    CameraImage,
    CameraIntrinsicsBrownConrady,
    CameraIntrinsicsPinhole,
    Frame,
    LidarCalibration,
    LidarRangeImage,
    SensorRig,
)


class WaymoFrameParser:
    """Parser for Waymo Open Dataset frames. Provides methods to parse camera images, lidar range images, and sensor calibrations from DataFrames loaded from the dataset."""

    def parse_frame(
        self,
        timestamp_micros: int,
        camera_images_df: pd.DataFrame,
        lidar_range_images_df: pd.DataFrame,
    ) -> Frame:
        """Parse a frame from camera images and lidar range images DataFrames.

        Args:
            timestamp_micros: The timestamp of the frame in microseconds.
            camera_images_df: DataFrame containing camera image data.
            lidar_range_images_df: DataFrame containing lidar range image data.

        Returns:
            A Frame object containing the parsed camera images and lidar range images.
        """
        camera_images = self.parse_camera_images(camera_images_df)
        lidar_range_images = self.parse_lidar_range_images(lidar_range_images_df)
        return Frame(
            timestamp_micros=timestamp_micros,
            camera_images=camera_images,
            lidar_range_images=lidar_range_images,
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
                camera_calibrations[Camera(camera_name)] = CameraCalibration(
                    camera_name=Camera(camera_name),
                    intrinsic_matrix=intrinsic_matrix,
                    extrinsic_matrix=extrinsic_matrix,
                )
        lidar_calibrations = {}
        if lidar_calibration_df is not None:
            for _, row in lidar_calibration_df.iterrows():
                lidar_name = row["key.laser_name"]
                extrinsic_matrix = self._parse_lidar_extrinsics(row)
                lidar_calibrations[Lidar(lidar_name)] = LidarCalibration(
                    lidar_name=Lidar(lidar_name),
                    extrinsic_matrix=extrinsic_matrix,
                )

        return SensorRig(cameras=camera_calibrations, lidars=lidar_calibrations)

    def parse_camera_images(
        self, camera_images_df: pd.DataFrame
    ) -> dict[Camera, CameraImage]:
        """Parse camera images from the DataFrame.

        Args:
            camera_images_df: DataFrame containing camera image data.

        Returns:
            A dictionary mapping Camera enum to CameraImage objects.
        """
        camera_images = {}
        for _, row in camera_images_df.iterrows():
            camera_name: int = row["key.camera_name"]
            timestamp_micros: int = row["key.frame_timestamp_micros"]
            image_data: bytes = row["[CameraImageComponent].image"]  # binary JPEG data
            decoded_image: np.ndarray = self._decode_jpeg(image_data)
            camera_images[Camera(camera_name)] = CameraImage(
                camera_name=Camera(camera_name),
                timestamp_micros=timestamp_micros,
                image=decoded_image,
            )
        return camera_images

    def parse_lidar_range_images(
        self, lidar_range_images_df: pd.DataFrame, returns: list[int] | None = None
    ) -> dict[Lidar, list[LidarRangeImage]]:
        """Parse lidar range images from the DataFrame.

        Args:
            lidar_range_images_df: DataFrame containing lidar range image data.
            returns: List of return counts to parse (e.g., [1, 2]). Defaults to [1].

        Returns:
            A dictionary mapping Lidar enum to a list of LidarRangeImage objects.
        """

        def _convert_range_image_to_nparray(
            range_image: list[float], shape: list[int]
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

        returns = returns or [1]  # default to return 1 if not specified
        lidar_range_images = {}
        for _, row in lidar_range_images_df.iterrows():
            lidar_name: int = row["key.laser_name"]
            timestamp_micros: int = row["key.frame_timestamp_micros"]
            lidar_returns = []
            if 1 in returns:
                range_image_1 = _convert_range_image_to_nparray(
                    row["[LidarComponent].range_image_return1.values"],
                    row["[LidarComponent].range_image_return1.shape"],
                )
                lidar_returns.append(
                    LidarRangeImage(
                        lidar_name=Lidar(lidar_name),
                        timestamp_micros=timestamp_micros,
                        range_image=range_image_1,
                        return_count=1,
                    )
                )
            if 2 in returns:
                range_image_2 = _convert_range_image_to_nparray(
                    row["[LidarComponent].range_image_return2.values"],
                    row["[LidarComponent].range_image_return2.shape"],
                )
                lidar_returns.append(
                    LidarRangeImage(
                        lidar_name=Lidar(lidar_name),
                        timestamp_micros=timestamp_micros,
                        range_image=range_image_2,
                        return_count=2,
                    )
                )
            if not lidar_returns:
                raise ValueError(
                    f"No valid returns found for lidar {lidar_name} at timestamp {timestamp_micros}. Check if the specified returns exist in the DataFrame."
                )
            lidar_range_images[Lidar(lidar_name)] = lidar_returns
        return lidar_range_images

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
            calibration_row["[LidarCalibrationComponent].extrinsic.transform"]
        )
