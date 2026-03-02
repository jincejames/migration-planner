from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime

import yaml

_MANDATORY_FIELDS = ("volume_name", "input_dependency_name")


@dataclass
class PlannerConfig:
    # Mandatory — must be supplied via YAML or CLI
    volume_name: str
    input_dependency_name: str

    # Optional — absent means the feature is not used
    outofscope_stream_file_name: str | None = None
    report_dependency_file_name: str | None = None
    table_size_file_name: str | None = None

    # --- mandatory derived paths ---

    @property
    def dependency_input_path(self) -> str:
        return self.volume_name + self.input_dependency_name

    @property
    def output_path(self) -> str:
        dir_name = "community_detection_output_" + datetime.now().strftime("%d%m%Y_%H")
        return self.volume_name + "community_detection_output_latest/" + dir_name + "/"

    @property
    def latest_path(self) -> str:
        return self.volume_name + "community_detection_output_latest/"

    # --- optional derived paths (None when file name not provided) ---

    @property
    def outofscope_stream_path(self) -> str | None:
        if self.outofscope_stream_file_name is None:
            return None
        return self.volume_name + self.outofscope_stream_file_name

    @property
    def report_dependency_path(self) -> str | None:
        if self.report_dependency_file_name is None:
            return None
        return self.volume_name + self.report_dependency_file_name

    @property
    def table_size_path(self) -> str | None:
        if self.table_size_file_name is None:
            return None
        return self.volume_name + self.table_size_file_name


def load_config(args: list[str] | None = None) -> PlannerConfig:
    parser = argparse.ArgumentParser(description="Migration Planner — Community Detection")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--volume-name")
    parser.add_argument("--input-dependency-name")
    parser.add_argument("--outofscope-stream-file-name")
    parser.add_argument("--report-dependency-file-name")
    parser.add_argument("--table-size-file-name")
    parsed = parser.parse_args(args)

    # Collect values — all start as absent (None)
    values: dict[str, str | None] = {
        "volume_name": None,
        "input_dependency_name": None,
        "outofscope_stream_file_name": None,
        "report_dependency_file_name": None,
        "table_size_file_name": None,
    }

    # Layer 1: YAML file
    if parsed.config:
        with open(parsed.config) as fh:
            data = yaml.safe_load(fh) or {}
        for key, value in data.items():
            if key in values:
                values[key] = value

    # Layer 2: CLI args (highest priority)
    cli_map = {
        "volume_name": parsed.volume_name,
        "input_dependency_name": parsed.input_dependency_name,
        "outofscope_stream_file_name": parsed.outofscope_stream_file_name,
        "report_dependency_file_name": parsed.report_dependency_file_name,
        "table_size_file_name": parsed.table_size_file_name,
    }
    for attr, value in cli_map.items():
        if value is not None:
            values[attr] = value

    # Validate mandatory fields
    missing = [k for k in _MANDATORY_FIELDS if values[k] is None]
    if missing:
        raise ValueError(f"Required config fields not provided: {', '.join(missing)}")

    return PlannerConfig(**values)
