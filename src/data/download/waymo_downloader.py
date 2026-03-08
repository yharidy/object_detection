import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from functools import partial

import click
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class DatasetSplit(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "testing"


class Stream(Enum):
    CAMERA = "camera"
    LIDAR = "lidar"


RAW_DATA_FOLDER = {"v_2_0_0": {Stream.CAMERA: "camera_image", Stream.LIDAR: "lidar"}}

CALIBRATION_FOLDER = {
    "v_2_0_0": {
        Stream.CAMERA: "camera_calibration",
        Stream.LIDAR: "lidar_calibration",
    }
}

GT_LABEL_FOLDER = {"v_2_0_0": {Stream.CAMERA: "camera_box", Stream.LIDAR: "lidar_box"}}


_thread_local = threading.local()


def get_client():
    def setup_client():
        try:
            client = storage.Client()
            return client
        except DefaultCredentialsError as e:
            raise e

    if not hasattr(_thread_local, "client"):
        _thread_local.client = setup_client()
    return _thread_local.client


def select_segments_from_folder(
    bucket: storage.Bucket, path_to_folder: str, num_segments: int
) -> list:
    selected_segments = [blob.name.split('/')[-1] for blob in bucket.list_blobs(prefix=path_to_folder, max_results=num_segments)]
    logger.info(
        f"Selected {len(selected_segments)} segments from folder {path_to_folder}."
    )
    return selected_segments



def download_file(bucket_name, folder, filename, output_dir):
    client = get_client()
    bucket = client.bucket(bucket_name)

    gcs_path = f"{folder}/{filename}"
    local_folder = os.path.join(output_dir, folder)
    os.makedirs(local_folder, exist_ok=True)

    local_path = os.path.join(local_folder, filename)

    if os.path.exists(local_path):
        logger.warning(f"File {local_path} already exists. Skipping download.")
        return

    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)
    return


@click.command()
@click.option(
    "--split",
    type=click.Choice([e.value for e in DatasetSplit]),
    default=DatasetSplit.TRAINING.value,
    help="Dataset split to download",
)
@click.option(
    "--streams",
    "-s",
    type=click.Choice([e.value for e in Stream]),
    multiple=True,
    default=[Stream.CAMERA.value],
    help="Streams to download",
)
@click.option(
    "--master-stream",
    type=click.Choice([e.value for e in Stream]),
    default=None,
    required=False,
    help="Stream to use for segment selection when --num-segments is specified. Defaults to first stream.",
)
@click.option(
    "--version",
    type=str,
    default="v_2_0_0",
    help="Version of the Waymo Open Dataset to download",
)
@click.option(
    "--num-segments",
    "-n",
    type=int,
    default=-1,
    help="Number of segments to download. Set to -1 to download all segments.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory to save downloaded segments",
)
@click.option(
    "--max-workers",
    type=int,
    default=16,
    help="Maximum number of parallel workers to use for downloading",
)
def main(split, streams, master_stream, version, num_segments, output_dir, max_workers):
    if output_dir is None:
        output_dir = os.path.join("data", "waymo", version, split)
    os.makedirs(output_dir, exist_ok=True)

    for stream in streams:
        if stream not in [e.value for e in Stream]:
            raise ValueError(f"Stream '{stream}' is not recognized. Valid options are: {[e.value for e in Stream]}")

    # set up client
    client = get_client()
    bucket_name = f"waymo_open_dataset_{version}"
    bucket = client.bucket(bucket_name)

    # Select segments to download
    if master_stream is None:
        master_stream = streams[0]
    logger.info(f"Using '{master_stream}' stream for segment selection.")
    selected_segments = []
    if num_segments > 0:
        raw_data_path = (
            f"{split}/{RAW_DATA_FOLDER.get(version, {}).get(Stream(master_stream))}/"
        )
        logger.info(f"Selecting {num_segments} segments from the dataset...")
        selected_segments = select_segments_from_folder(
            bucket, raw_data_path, num_segments
        )

    # Download blobs in parallel
    folders_to_download = []
    for stream in streams:
        folders_to_download.append(f"{split}/{RAW_DATA_FOLDER.get(version, {}).get(Stream(stream))}")
        folders_to_download.append(f"{split}/{CALIBRATION_FOLDER.get(version, {}).get(Stream(stream))}")
        folders_to_download.append(f"{split}/{GT_LABEL_FOLDER.get(version, {}).get(Stream(stream))}")
    
    files_to_download = [
        (folder, segment)
        for segment in selected_segments
        for folder in folders_to_download
    ]
    
    logger.info(f"Total files to download: {len(files_to_download)}")
    logger.info(f"Segments selected: {len(selected_segments)}, Folders: {len(folders_to_download)}")
    
    if len(files_to_download) == 0:
        logger.warning("No files to download. Check if segments were selected correctly.")
        return
    
    download_worker = partial(
        download_file, bucket_name=bucket_name, output_dir=output_dir
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [
            executor.submit(download_worker, folder=folder, filename=file)
            for folder, file in files_to_download
        ]

        for future in tqdm(
            as_completed(results),
            total=len(files_to_download),
            desc="Downloading files",
        ):
            try:
                future.result()  # This will raise any exceptions that occurred in the worker thread
            except Exception as e:
                logger.error(f"Download failed with error: {e}", exc_info=True)
    logger.info("Download completed.")


if __name__ == "__main__":
    main()
