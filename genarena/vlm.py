# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""VLM API calling module with retry and multi-endpoint support."""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Union
from urllib.parse import urlparse

from openai import OpenAI


logger = logging.getLogger(__name__)

def _silence_http_client_logs() -> None:
    """Silence noisy HTTP client logs (e.g., httpx INFO request lines)."""
    try:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    except Exception:
        pass


def _short_endpoint(base_url: str) -> str:
    """Convert base_url to a short host:port label for progress display."""
    if not base_url:
        return "default"
    try:
        u = urlparse(base_url)
        host = u.hostname or base_url
        port = f":{u.port}" if u.port else ""
        return f"{host}{port}"
    except Exception:
        return base_url


def _progress_put(progress: Any, item: Any) -> None:
    """Best-effort put to a queue-like progress sink."""
    if progress is None:
        return
    if hasattr(progress, "put"):
        try:
            progress.put(item)
        except Exception:
            pass


@dataclass
class EndpointConfig:
    """Configuration for an API endpoint."""

    base_url: str
    api_keys: list[str]
    timeout: int = 120
    max_retries: int = 3
    weight: float = 1.0
    disabled: bool = False


class MultiEndpointManager:
    """
    Manages multiple API endpoints with automatic failover and load balancing.

    Supports:
    - Round-robin endpoint selection
    - Per-endpoint API key rotation
    - Automatic endpoint disable/re-enable on errors
    - Thread-safe operations
    """

    def __init__(self, endpoint_configs: list[EndpointConfig], timeout: int = 120):
        """
        Initialize multi-endpoint manager.

        Args:
            endpoint_configs: List of endpoint configurations
            timeout: Default timeout for API calls
        """
        self.timeout = timeout
        self.endpoints: list[dict[str, Any]] = []

        for config in endpoint_configs:
            if not config.api_keys:
                continue

            self.endpoints.append({
                "config": config,
                "key_index": 0,  # Current API key index for round-robin
                "error_count": 0,
                "last_error_time": 0.0,
                "disabled": config.disabled,
            })

        if not self.endpoints:
            raise ValueError("At least one valid endpoint configuration is required")

        self.current_endpoint_index = 0
        self.lock = threading.Lock()

    def get_client(self) -> tuple[OpenAI, EndpointConfig, str]:
        """
        Get a client from available endpoints (thread-safe round-robin).

        Returns:
            Tuple of (OpenAI client, EndpointConfig, api_key used)

        Raises:
            RuntimeError: If all endpoints are disabled
        """
        with self.lock:
            # Check if any endpoints are available
            if not any(not ep["disabled"] for ep in self.endpoints):
                # Try to re-enable endpoints that have been disabled for > 5 minutes
                for ep in self.endpoints:
                    if ep["disabled"] and time.time() - ep["last_error_time"] > 300:
                        ep["disabled"] = False
                        ep["error_count"] = 0

                if not any(not ep["disabled"] for ep in self.endpoints):
                    raise RuntimeError("All endpoints are temporarily disabled")

            attempts = 0
            while attempts < len(self.endpoints):
                endpoint = self.endpoints[self.current_endpoint_index]
                self.current_endpoint_index = (self.current_endpoint_index + 1) % len(self.endpoints)

                # Skip disabled endpoints
                if endpoint["disabled"]:
                    # Check if we should re-enable
                    if time.time() - endpoint["last_error_time"] > 300:
                        endpoint["disabled"] = False
                        endpoint["error_count"] = 0
                    else:
                        attempts += 1
                        continue

                config = endpoint["config"]

                # Get API key with round-robin
                api_key = config.api_keys[endpoint["key_index"]]
                endpoint["key_index"] = (endpoint["key_index"] + 1) % len(config.api_keys)

                try:
                    client = OpenAI(
                        api_key=api_key,
                        base_url=config.base_url,
                        timeout=config.timeout or self.timeout,
                    )
                    return client, config, api_key
                except Exception as e:
                    logger.warning(f"Failed to create client for {config.base_url}: {e}")
                    attempts += 1

            raise RuntimeError("No available endpoints")

    def record_success(self, config: EndpointConfig) -> None:
        """
        Record a successful API call for an endpoint.

        Args:
            config: The endpoint config that succeeded
        """
        with self.lock:
            for endpoint in self.endpoints:
                if endpoint["config"].base_url == config.base_url:
                    # Reduce error count on success
                    if endpoint["error_count"] > 0:
                        endpoint["error_count"] = max(0, endpoint["error_count"] - 1)
                    # Re-enable if disabled
                    if endpoint["disabled"]:
                        endpoint["disabled"] = False
                    break

    def record_failure(self, config: EndpointConfig, error_type: str = "generic") -> None:
        """
        Record a failed API call for an endpoint.

        Args:
            config: The endpoint config that failed
            error_type: Type of error ('auth', 'rate_limit', 'timeout', 'generic')
        """
        with self.lock:
            for endpoint in self.endpoints:
                if endpoint["config"].base_url == config.base_url:
                    endpoint["error_count"] += 1
                    endpoint["last_error_time"] = time.time()

                    # Disable endpoint after 3 consecutive errors
                    if endpoint["error_count"] >= 3:
                        endpoint["disabled"] = True
                        logger.warning(
                            f"Endpoint {config.base_url} disabled after {endpoint['error_count']} errors"
                        )
                    break

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about endpoint usage."""
        with self.lock:
            stats = {
                "total_endpoints": len(self.endpoints),
                "enabled_endpoints": sum(1 for ep in self.endpoints if not ep["disabled"]),
                "disabled_endpoints": sum(1 for ep in self.endpoints if ep["disabled"]),
                "endpoints": [],
            }

            for endpoint in self.endpoints:
                stats["endpoints"].append({
                    "base_url": endpoint["config"].base_url,
                    "enabled": not endpoint["disabled"],
                    "error_count": endpoint["error_count"],
                    "num_keys": len(endpoint["config"].api_keys),
                })

            return stats


class VLMJudge:
    """
    VLM Judge class for calling vision-language models via OpenAI-compatible API.

    Supports:
    - Greedy mode (temperature=0) for reproducible results
    - Multi-endpoint with automatic failover
    - Per-endpoint API key rotation
    - Exponential backoff retry mechanism
    """

    DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        timeout: int = 120,
        max_retries: int = 3,
        base_url: Optional[str] = None,
        base_urls: Optional[Union[str, list[str]]] = None,
        api_key: Optional[str] = None,
        api_keys: Optional[Union[str, list[str]]] = None,
        progress: Any = None,
    ):
        """
        Initialize the VLM Judge.

        Args:
            model: Model name to use for evaluation
            temperature: Sampling temperature (0 for greedy/deterministic)
            timeout: API call timeout in seconds
            max_retries: Maximum number of retry attempts
            base_url: Single OpenAI API base URL (legacy, use base_urls for multi-endpoint)
            base_urls: Multiple base URLs (comma-separated string or list)
            api_key: Single API key (legacy, use api_keys for multiple)
            api_keys: Multiple API keys (comma-separated string or list)
        """
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._progress = progress

        # Avoid printing per-request httpx INFO lines to stdout.
        _silence_http_client_logs()

        # Parse base URLs
        self._base_urls = self._parse_urls(base_urls, base_url, "OPENAI_BASE_URLS", "OPENAI_BASE_URL")

        # Parse API keys
        self._api_keys = self._parse_keys(api_keys, api_key, "OPENAI_API_KEY")

        if not self._api_keys:
            raise ValueError(
                "API key must be provided via api_key/api_keys parameter or "
                "OPENAI_API_KEY environment variable"
            )

        # If progress is enabled, silence noisy per-request httpx logs.
        if self._progress is not None:
            _silence_http_client_logs()

        # Build endpoint configs
        endpoint_configs = self._build_endpoint_configs()

        # Initialize multi-endpoint manager
        self.endpoint_manager = MultiEndpointManager(endpoint_configs, timeout=timeout)

        # For backward compatibility and logging
        self.base_url = self._base_urls[0] if self._base_urls else None
        self.api_key = self._api_keys[0] if self._api_keys else None

    def set_progress(self, progress: Any) -> None:
        """Attach a progress sink (queue-like) for emitting request events."""
        self._progress = progress
        if self._progress is not None:
            _silence_http_client_logs()

    def _progress_event(self, msg: str) -> None:
        """Emit a short request event for progress UI."""
        _progress_put(self._progress, ("log", msg))

    def _parse_urls(
        self,
        urls: Optional[Union[str, list[str]]],
        single_url: Optional[str],
        env_multi: str,
        env_single: str
    ) -> list[str]:
        """Parse base URLs from various sources."""
        # Priority: urls param > single_url param > env vars
        if urls:
            if isinstance(urls, str):
                return [u.strip() for u in urls.split(",") if u.strip()]
            return list(urls)

        if single_url:
            return [single_url]

        # Try environment variables
        env_urls = os.environ.get(env_multi) or os.environ.get(env_single)
        if env_urls:
            return [u.strip() for u in env_urls.split(",") if u.strip()]

        return []

    def _parse_keys(
        self,
        keys: Optional[Union[str, list[str]]],
        single_key: Optional[str],
        env_name: str
    ) -> list[str]:
        """Parse API keys from various sources."""
        # Priority: keys param > single_key param > env var
        if keys:
            if isinstance(keys, str):
                return [k.strip() for k in keys.split(",") if k.strip()]
            return list(keys)

        if single_key:
            return [single_key]

        # Try environment variable
        env_keys = os.environ.get(env_name)
        if env_keys:
            return [k.strip() for k in env_keys.split(",") if k.strip()]

        return []

    def _build_endpoint_configs(self) -> list[EndpointConfig]:
        """Build endpoint configurations from URLs and keys."""
        configs = []

        if not self._base_urls:
            # No base URL specified, create single endpoint with all keys
            configs.append(EndpointConfig(
                base_url="",  # Will use OpenAI default
                api_keys=self._api_keys,
                timeout=self.timeout,
                max_retries=self.max_retries,
            ))
        else:
            # Distribute API keys among endpoints
            num_urls = len(self._base_urls)
            num_keys = len(self._api_keys)
            keys_per_endpoint = max(1, num_keys // num_urls)
            remainder = num_keys % num_urls

            key_index = 0
            for i, url in enumerate(self._base_urls):
                # Calculate keys for this endpoint
                num_keys_for_endpoint = keys_per_endpoint
                if i < remainder:
                    num_keys_for_endpoint += 1

                endpoint_keys = self._api_keys[key_index:key_index + num_keys_for_endpoint]
                if not endpoint_keys:
                    # If no keys left, use all keys
                    endpoint_keys = self._api_keys
                key_index += num_keys_for_endpoint

                configs.append(EndpointConfig(
                    base_url=url,
                    api_keys=endpoint_keys,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                ))

        return configs

    def call(self, messages: list[dict[str, Any]]) -> str:
        """
        Call the VLM API and return the response text.

        Uses multi-endpoint failover and exponential backoff for retries.

        Args:
            messages: List of message dicts in OpenAI Chat Completion format

        Returns:
            Raw response text from the VLM

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Get client from endpoint manager
                client, config, _ = self.endpoint_manager.get_client()
                endpoint_label = _short_endpoint(getattr(config, "base_url", "") or "")

                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=4096,
                )

                # Extract text from response
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content:
                        # Record success
                        self.endpoint_manager.record_success(config)
                        self._progress_event(f"OK {endpoint_label}")
                        return content
                    else:
                        raise ValueError("Empty response content from VLM")
                else:
                    raise ValueError("No choices in VLM response")

            except Exception as e:
                last_exception = e
                endpoint_label = _short_endpoint(getattr(config, "base_url", "") or "") if "config" in dir() else "unknown"

                # Record failure for endpoint
                if 'config' in dir():
                    error_type = self._classify_error(e)
                    self.endpoint_manager.record_failure(config, error_type)
                    self._progress_event(f"ERR {endpoint_label} {error_type}")
                else:
                    self._progress_event(f"ERR {endpoint_label} generic")

                logger.warning(
                    f"VLM call attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )

                # Exponential backoff: 1s, 2s, 4s, ...
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        # All retries failed
        raise Exception(
            f"VLM call failed after {self.max_retries} attempts. "
            f"Last error: {last_exception}"
        )

    def _classify_error(self, error: Exception) -> str:
        """Classify an error for endpoint management."""
        error_str = str(error).lower()

        if "401" in error_str or "unauthorized" in error_str or "invalid api key" in error_str:
            return "auth"
        elif "429" in error_str or "rate limit" in error_str:
            return "rate_limit"
        elif "timeout" in error_str:
            return "timeout"
        else:
            return "generic"

    def call_with_raw(
        self,
        messages: list[dict[str, Any]]
    ) -> tuple[str, Optional[Exception]]:
        """
        Call the VLM API and return both response and any error.

        This variant returns the error instead of raising it,
        useful for audit logging.

        Args:
            messages: List of message dicts in OpenAI Chat Completion format

        Returns:
            Tuple of (response_text, error) where error is None on success
        """
        try:
            response = self.call(messages)
            return response, None
        except Exception as e:
            return "", e

    def get_endpoint_stats(self) -> dict[str, Any]:
        """Get statistics about endpoint usage."""
        return self.endpoint_manager.get_stats()

    @property
    def config(self) -> dict[str, Any]:
        """
        Get the judge configuration for logging/persistence.

        Returns:
            Dict with model, temperature, timeout, max_retries, and endpoint info
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "base_urls": self._base_urls,
            "num_api_keys": len(self._api_keys),
            "endpoint_stats": self.get_endpoint_stats(),
        }
