from __future__ import annotations
import os
import requests
from typing import Optional


class ImageRoutes:
    def __init__(self, root:CocoAnnotator) -> None:
        self.root = root
        self.base_url = os.path.join(self.root.base_url, "image")

    def filter(self, per_page: Optional[int]=1000, limit:Optional[int]=None) -> list[dict]:
        url = os.path.join(self.base_url)
        page = 1
        has_more = True
        images = []
        while has_more:
            req = requests.get(url, params={"page": page, "per_page": per_page}, cookies=self.root.cookies)
            req.raise_for_status()
            chunk_images = req.json()["images"]
            if len(chunk_images) == 0:
                has_more = False
                break
            images += chunk_images
            page += 1
            if limit is not None and len(images) >= limit:
                images = images[:limit]
                break
        return images

class CocoAnnotator:
    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or "http://annotator:5000/api"
        self.cookies = None

    def login(self, username: str, password: str) -> requests.Response:
        url = os.path.join(self.base_url, "user/login")
        req = requests.post(url, json={"username": username, "password": password})
        req.raise_for_status()
        self.cookies = req.cookies
        return req.json()

    @property
    def image(self) -> ImageRoutes:
        return ImageRoutes(self)
