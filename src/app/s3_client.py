from __future__ import annotations

import os
from typing import Iterable, List, Dict, Optional

import boto3
from botocore.config import Config
from loguru import logger


class WasabiClient:
    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        session_kwargs = {}
        if profile:
            session_kwargs["profile_name"] = profile
        self.session = boto3.Session(**session_kwargs)
        endpoint_url = f"https://s3.{region}.wasabisys.com"
        self.s3 = self.session.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            config=Config(s3={"addressing_style": "virtual"}),
        )

    def list_objects(self, bucket: str, prefix: str = "") -> List[Dict]:
        paginator = self.s3.get_paginator("list_objects_v2")
        results: List[Dict] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                results.append(obj)
        return results


    def download_object(self, bucket: str, key: str, dest_dir: str) -> str:
        os.makedirs(dest_dir, exist_ok=True)
        filename = key.split("/")[-1] or "download"
        dest_path = os.path.join(dest_dir, filename)
        logger.info("Downloading s3://{}/{} -> {}", bucket, key, dest_path)
        self.s3.download_file(bucket, key, dest_path)
        return dest_path

    def upload_file(self, bucket: str, path: str, key: str) -> None:
        logger.info("Uploading {} -> s3://{}/{}", path, bucket, key)
        self.s3.upload_file(path, bucket, key, ExtraArgs={"ContentType": "video/mp4"})


