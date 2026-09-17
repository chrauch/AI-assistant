# This file is part of AIA. Copyright (C) 2026 Christian Rauch.
# Distributed under terms of the GPL3 license.

from ipaddress import ip_address
from importlib import import_module
import socket
from typing import Any
from urllib.parse import urlparse

from .base import Tool


class FetchWebpageTool(Tool):
    name = "fetch_webpage"
    description = (
        "Fetch a public webpage and return its main text content without HTML "
        "markup, navigation, or most boilerplate."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "A webpage URL.",
            }
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def run(self, url: str, **_: Any) -> str:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise ValueError("URL must use http or https and include a hostname")
        self._reject_private_host(parsed_url.hostname)

        try:
            trafilatura = import_module("trafilatura")
        except ImportError as error:
            raise ValueError(
                "The fetch_webpage tool requires the trafilatura package"
            ) from error

        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                raise ValueError("Could not download webpage")
            text = trafilatura.extract(
                downloaded,
                include_links=True,
                include_tables=True,
            )
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"Could not fetch webpage: {error}") from error
        if not text:
            raise ValueError("Could not extract readable text from webpage")
        return text

    @staticmethod
    def _reject_private_host(hostname: str) -> None:
        if hostname.lower() == "localhost":
            raise ValueError("Localhost URLs are not allowed")
        try:
            addresses = {
                ip_address(hostname),
            }
        except ValueError:
            try:
                addresses = {
                    ip_address(address[4][0])
                    for address in socket.getaddrinfo(hostname, None)
                }
            except OSError as error:
                raise ValueError(f"Could not resolve URL hostname: {hostname}") from error
        if any(
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            for address in addresses
        ):
            raise ValueError("Private or local network URLs are not allowed")
