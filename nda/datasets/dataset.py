#!/usr/bin/env python
# coding=utf-8
import os
import logging
from urllib import request

logger = logging.getLogger(__name__)

class Dataset(object):
    def __init__(self):
        self.data_dir = os.path.expanduser('~/data')

    def download_file(self, url, local_path):
        """下载文件的通用方法"""
        if not os.path.exists(local_path):
            logger.info(f"Downloading from {url} to {local_path}")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                request.urlretrieve(url, local_path)
                logger.info("Download completed")
            except Exception as e:
                logger.error(f"Download failed: {str(e)}")
                raise
