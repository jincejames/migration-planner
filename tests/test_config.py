"""
Comprehensive tests for migration_planner.utils.config.

Contract under test:
  - volume_name and input_dependency_name are MANDATORY — must come from YAML or CLI
  - outofscope_stream_file_name, report_dependency_file_name, table_size_file_name
    are OPTIONAL — absent means the derived path property returns None
  - load_config() raises ValueError if any mandatory field is missing after
    merging all sources; raises FileNotFoundError / yaml.YAMLError for bad files
  - Priority chain: CLI > YAML (no hardcoded defaults)

Coverage:
  - PlannerConfig construction (mandatory required, optional default to None)
  - PlannerConfig mutability and dynamic property behaviour
  - PlannerConfig dataclass equality
  - All derived-path properties — mandatory always str, optional str | None
  - output_path with frozen clock
  - load_config() validation of mandatory fields
  - load_config() CLI arguments — each flag, all flags, partial flags
  - load_config() YAML loading — full, partial, unknown keys, empty file, project config.yaml
  - Priority chain: CLI > YAML
  - Error cases: missing mandatory fields, missing file, invalid YAML, unknown CLI flag
  - Integration: leiden.py variable-assignment pattern
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from migration_planner.utils.config import PlannerConfig, load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOLUME = "/test/volume/"
DEP_NAME = "deps.csv"

FIXED_DT = datetime(2025, 6, 15, 9, 0, 0)
FIXED_DIR_NAME = "community_detection_output_15062025_09"


def minimal(**overrides) -> PlannerConfig:
    """Return a PlannerConfig with the two mandatory fields set, plus any overrides."""
    kwargs = {"volume_name": VOLUME, "input_dependency_name": DEP_NAME}
    kwargs.update(overrides)
    return PlannerConfig(**kwargs)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def yaml_file(tmp_path):
    """Return a helper that writes a dict as YAML and returns the file path."""

    def _write(content: dict) -> str:
        fpath = tmp_path / "config.yaml"
        fpath.write_text(yaml.dump(content))
        return str(fpath)

    return _write


# ---------------------------------------------------------------------------
# PlannerConfig — construction contract
# ---------------------------------------------------------------------------


class TestPlannerConfigConstruction:
    def test_mandatory_fields_accepted(self):
        cfg = PlannerConfig(volume_name="/v/", input_dependency_name="dep.csv")
        assert cfg.volume_name == "/v/"
        assert cfg.input_dependency_name == "dep.csv"

    def test_optional_fields_default_to_none(self):
        cfg = PlannerConfig(volume_name="/v/", input_dependency_name="dep.csv")
        assert cfg.outofscope_stream_file_name is None
        assert cfg.report_dependency_file_name is None
        assert cfg.table_size_file_name is None

    def test_missing_volume_name_raises_type_error(self):
        with pytest.raises(TypeError):
            PlannerConfig(input_dependency_name="dep.csv")  # type: ignore[call-arg]

    def test_missing_input_dependency_name_raises_type_error(self):
        with pytest.raises(TypeError):
            PlannerConfig(volume_name="/v/")  # type: ignore[call-arg]

    def test_missing_both_mandatory_raises_type_error(self):
        with pytest.raises(TypeError):
            PlannerConfig()  # type: ignore[call-arg]

    def test_all_optional_fields_accepted(self):
        cfg = PlannerConfig(
            volume_name="/v/",
            input_dependency_name="dep.csv",
            outofscope_stream_file_name="oos.csv",
            report_dependency_file_name="rep.csv",
            table_size_file_name="size.csv",
        )
        assert cfg.outofscope_stream_file_name == "oos.csv"
        assert cfg.report_dependency_file_name == "rep.csv"
        assert cfg.table_size_file_name == "size.csv"


# ---------------------------------------------------------------------------
# PlannerConfig — mutability
# ---------------------------------------------------------------------------


class TestPlannerConfigMutability:
    def test_volume_name_can_be_reassigned(self):
        cfg = minimal()
        cfg.volume_name = "/new/"
        assert cfg.volume_name == "/new/"

    def test_mandatory_path_properties_reflect_reassigned_volume_name(self):
        cfg = minimal()
        cfg.volume_name = "/new/"
        assert cfg.dependency_input_path.startswith("/new/")
        assert cfg.latest_path.startswith("/new/")

    def test_optional_path_property_reflects_reassigned_volume_name(self):
        cfg = minimal(outofscope_stream_file_name="oos.csv")
        cfg.volume_name = "/new/"
        assert cfg.outofscope_stream_path == "/new/oos.csv"

    def test_input_dependency_name_reassignment_updates_path(self):
        cfg = minimal()
        cfg.input_dependency_name = "updated.csv"
        assert cfg.dependency_input_path == VOLUME + "updated.csv"

    def test_optional_field_set_after_init_updates_path(self):
        cfg = minimal()
        assert cfg.outofscope_stream_path is None
        cfg.outofscope_stream_file_name = "oos.csv"
        assert cfg.outofscope_stream_path == VOLUME + "oos.csv"


# ---------------------------------------------------------------------------
# PlannerConfig — dataclass equality
# ---------------------------------------------------------------------------


class TestPlannerConfigEquality:
    def test_identical_mandatory_fields_are_equal(self):
        assert minimal() == minimal()

    def test_different_volume_names_are_not_equal(self):
        assert PlannerConfig(volume_name="/a/", input_dependency_name="d.csv") != \
               PlannerConfig(volume_name="/b/", input_dependency_name="d.csv")

    def test_different_input_dependency_names_are_not_equal(self):
        assert PlannerConfig(volume_name="/v/", input_dependency_name="a.csv") != \
               PlannerConfig(volume_name="/v/", input_dependency_name="b.csv")

    def test_same_optional_fields_are_equal(self):
        a = minimal(outofscope_stream_file_name="oos.csv")
        b = minimal(outofscope_stream_file_name="oos.csv")
        assert a == b

    def test_different_optional_field_values_not_equal(self):
        assert minimal(outofscope_stream_file_name="a.csv") != \
               minimal(outofscope_stream_file_name="b.csv")

    def test_none_vs_set_optional_field_not_equal(self):
        assert minimal() != minimal(outofscope_stream_file_name="oos.csv")


# ---------------------------------------------------------------------------
# PlannerConfig — mandatory derived-path properties
# ---------------------------------------------------------------------------


class TestPlannerConfigMandatoryPaths:
    def test_dependency_input_path_concatenates_fields(self):
        cfg = PlannerConfig(volume_name="/base/", input_dependency_name="dep.csv")
        assert cfg.dependency_input_path == "/base/dep.csv"

    def test_latest_path_appends_fixed_suffix(self):
        cfg = PlannerConfig(volume_name="/base/", input_dependency_name="dep.csv")
        assert cfg.latest_path == "/base/community_detection_output_latest/"

    def test_dependency_input_path_updates_on_volume_change(self):
        cfg = minimal()
        cfg.volume_name = "/changed/"
        assert cfg.dependency_input_path == "/changed/" + DEP_NAME

    def test_latest_path_updates_on_volume_change(self):
        cfg = minimal()
        cfg.volume_name = "/changed/"
        assert cfg.latest_path == "/changed/community_detection_output_latest/"


# ---------------------------------------------------------------------------
# PlannerConfig — optional derived-path properties
# ---------------------------------------------------------------------------


class TestPlannerConfigOptionalPaths:
    def test_outofscope_stream_path_is_none_when_field_absent(self):
        assert minimal().outofscope_stream_path is None

    def test_report_dependency_path_is_none_when_field_absent(self):
        assert minimal().report_dependency_path is None

    def test_table_size_path_is_none_when_field_absent(self):
        assert minimal().table_size_path is None

    def test_outofscope_stream_path_when_field_set(self):
        cfg = minimal(outofscope_stream_file_name="oos.csv")
        assert cfg.outofscope_stream_path == VOLUME + "oos.csv"

    def test_report_dependency_path_when_field_set(self):
        cfg = minimal(report_dependency_file_name="rep.csv")
        assert cfg.report_dependency_path == VOLUME + "rep.csv"

    def test_table_size_path_when_field_set(self):
        cfg = minimal(table_size_file_name="size.csv")
        assert cfg.table_size_path == VOLUME + "size.csv"

    def test_all_optional_paths_when_all_fields_set(self):
        cfg = PlannerConfig(
            volume_name="/base/",
            input_dependency_name="dep.csv",
            outofscope_stream_file_name="oos.csv",
            report_dependency_file_name="rep.csv",
            table_size_file_name="size.csv",
        )
        assert cfg.outofscope_stream_path == "/base/oos.csv"
        assert cfg.report_dependency_path == "/base/rep.csv"
        assert cfg.table_size_path == "/base/size.csv"

    def test_optional_path_updates_on_volume_change(self):
        cfg = minimal(outofscope_stream_file_name="oos.csv")
        cfg.volume_name = "/changed/"
        assert cfg.outofscope_stream_path == "/changed/oos.csv"

    def test_optional_path_transitions_none_to_value(self):
        cfg = minimal()
        assert cfg.report_dependency_path is None
        cfg.report_dependency_file_name = "rep.csv"
        assert cfg.report_dependency_path == VOLUME + "rep.csv"


# ---------------------------------------------------------------------------
# PlannerConfig — output_path (datetime-dependent)
# ---------------------------------------------------------------------------


class TestPlannerConfigOutputPath:
    @patch("migration_planner.utils.config.datetime")
    def test_output_path_exact_value(self, mock_dt):
        mock_dt.now.return_value = FIXED_DT
        cfg = PlannerConfig(volume_name="/base/", input_dependency_name="dep.csv")
        assert cfg.output_path == f"/base/community_detection_output_latest/{FIXED_DIR_NAME}/"

    @patch("migration_planner.utils.config.datetime")
    def test_output_path_uses_volume_name(self, mock_dt):
        mock_dt.now.return_value = FIXED_DT
        cfg = minimal(volume_name="/custom/vol/")
        assert cfg.output_path.startswith("/custom/vol/")

    @patch("migration_planner.utils.config.datetime")
    def test_output_path_ends_with_slash(self, mock_dt):
        mock_dt.now.return_value = FIXED_DT
        assert minimal().output_path.endswith("/")

    @patch("migration_planner.utils.config.datetime")
    def test_output_path_contains_dir_name(self, mock_dt):
        mock_dt.now.return_value = FIXED_DT
        assert FIXED_DIR_NAME in minimal().output_path

    @patch("migration_planner.utils.config.datetime")
    def test_output_path_reflects_hour(self, mock_dt):
        mock_dt.now.return_value = datetime(2025, 6, 15, 14, 0, 0)
        assert "14" in minimal().output_path

    @patch("migration_planner.utils.config.datetime")
    def test_output_path_nested_under_latest_path(self, mock_dt):
        mock_dt.now.return_value = FIXED_DT
        cfg = minimal()
        assert cfg.output_path.startswith(cfg.latest_path)
        assert len(cfg.output_path) > len(cfg.latest_path)

    @patch("migration_planner.utils.config.datetime")
    def test_output_path_updates_on_volume_name_change(self, mock_dt):
        mock_dt.now.return_value = FIXED_DT
        cfg = minimal()
        cfg.volume_name = "/changed/"
        assert cfg.output_path.startswith("/changed/")

    @patch("migration_planner.utils.config.datetime")
    def test_output_path_strftime_ddmmyyyy_hh_format(self, mock_dt):
        mock_dt.now.return_value = datetime(2024, 1, 5, 8, 0, 0)
        assert "community_detection_output_05012024_08" in minimal().output_path


# ---------------------------------------------------------------------------
# load_config — mandatory field validation
# ---------------------------------------------------------------------------


class TestLoadConfigValidation:
    def test_missing_both_mandatory_fields_raises_value_error(self):
        with pytest.raises(ValueError, match="volume_name"):
            load_config([])

    def test_missing_volume_name_raises_value_error(self):
        with pytest.raises(ValueError, match="volume_name"):
            load_config(["--input-dependency-name", "dep.csv"])

    def test_missing_input_dependency_name_raises_value_error(self):
        with pytest.raises(ValueError, match="input_dependency_name"):
            load_config(["--volume-name", "/v/"])

    def test_error_message_lists_all_missing_fields(self):
        with pytest.raises(ValueError) as exc_info:
            load_config([])
        msg = str(exc_info.value)
        assert "volume_name" in msg
        assert "input_dependency_name" in msg

    def test_both_mandatory_fields_provided_via_cli_succeeds(self):
        cfg = load_config(["--volume-name", "/v/", "--input-dependency-name", "dep.csv"])
        assert isinstance(cfg, PlannerConfig)

    def test_none_args_with_empty_sys_argv_raises_value_error(self):
        with patch.object(sys, "argv", ["prog"]):
            with pytest.raises(ValueError):
                load_config(None)


# ---------------------------------------------------------------------------
# load_config — CLI arguments
# ---------------------------------------------------------------------------


class TestLoadConfigCLIArgs:
    BASE = ["--volume-name", "/v/", "--input-dependency-name", "dep.csv"]

    def test_volume_name_flag(self):
        cfg = load_config(["--volume-name", "/cli/", "--input-dependency-name", "dep.csv"])
        assert cfg.volume_name == "/cli/"

    def test_input_dependency_name_flag(self):
        cfg = load_config(["--volume-name", "/v/", "--input-dependency-name", "cli.csv"])
        assert cfg.input_dependency_name == "cli.csv"

    def test_outofscope_stream_file_name_flag(self):
        cfg = load_config(self.BASE + ["--outofscope-stream-file-name", "oos.csv"])
        assert cfg.outofscope_stream_file_name == "oos.csv"

    def test_report_dependency_file_name_flag(self):
        cfg = load_config(self.BASE + ["--report-dependency-file-name", "rep.csv"])
        assert cfg.report_dependency_file_name == "rep.csv"

    def test_table_size_file_name_flag(self):
        cfg = load_config(self.BASE + ["--table-size-file-name", "sz.csv"])
        assert cfg.table_size_file_name == "sz.csv"

    def test_all_flags_together(self):
        cfg = load_config([
            "--volume-name", "/v/",
            "--input-dependency-name", "d.csv",
            "--outofscope-stream-file-name", "o.csv",
            "--report-dependency-file-name", "r.csv",
            "--table-size-file-name", "s.csv",
        ])
        assert cfg.volume_name == "/v/"
        assert cfg.input_dependency_name == "d.csv"
        assert cfg.outofscope_stream_file_name == "o.csv"
        assert cfg.report_dependency_file_name == "r.csv"
        assert cfg.table_size_file_name == "s.csv"

    def test_unspecified_optional_flags_remain_none(self):
        cfg = load_config(self.BASE)
        assert cfg.outofscope_stream_file_name is None
        assert cfg.report_dependency_file_name is None
        assert cfg.table_size_file_name is None

    def test_cli_volume_name_propagates_to_dependency_input_path(self):
        cfg = load_config(["--volume-name", "/tmp/", "--input-dependency-name", "dep.csv"])
        assert cfg.dependency_input_path == "/tmp/dep.csv"

    def test_cli_optional_flag_produces_path(self):
        cfg = load_config(self.BASE + ["--outofscope-stream-file-name", "oos.csv"])
        assert cfg.outofscope_stream_path == "/v/oos.csv"

    def test_cli_omitted_optional_flag_path_is_none(self):
        cfg = load_config(self.BASE)
        assert cfg.outofscope_stream_path is None

    def test_returns_planner_config_instance(self):
        assert isinstance(load_config(self.BASE), PlannerConfig)


# ---------------------------------------------------------------------------
# load_config — YAML file
# ---------------------------------------------------------------------------


class TestLoadConfigYAML:
    MANDATORY = {"volume_name": "/yaml/vol/", "input_dependency_name": "yaml-dep.csv"}

    def test_all_fields_from_yaml(self, yaml_file):
        path = yaml_file({
            **self.MANDATORY,
            "outofscope_stream_file_name": "yaml-oos.csv",
            "report_dependency_file_name": "yaml-rep.csv",
            "table_size_file_name": "yaml-size.csv",
        })
        cfg = load_config(["--config", path])
        assert cfg.volume_name == "/yaml/vol/"
        assert cfg.input_dependency_name == "yaml-dep.csv"
        assert cfg.outofscope_stream_file_name == "yaml-oos.csv"
        assert cfg.report_dependency_file_name == "yaml-rep.csv"
        assert cfg.table_size_file_name == "yaml-size.csv"

    def test_yaml_with_only_mandatory_fields_leaves_optionals_none(self, yaml_file):
        path = yaml_file(self.MANDATORY)
        cfg = load_config(["--config", path])
        assert cfg.outofscope_stream_file_name is None
        assert cfg.report_dependency_file_name is None
        assert cfg.table_size_file_name is None

    def test_yaml_missing_mandatory_raises_value_error(self, yaml_file):
        path = yaml_file({"outofscope_stream_file_name": "oos.csv"})
        with pytest.raises(ValueError):
            load_config(["--config", path])

    def test_yaml_unknown_keys_are_silently_ignored(self, yaml_file):
        path = yaml_file({**self.MANDATORY, "unknown_key": "ignored", "also_unknown": 42})
        cfg = load_config(["--config", path])
        assert cfg.volume_name == "/yaml/vol/"
        assert not hasattr(cfg, "unknown_key")

    def test_empty_yaml_file_raises_value_error(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ValueError):
            load_config(["--config", str(empty)])

    def test_yaml_with_only_comments_raises_value_error(self, tmp_path):
        comments_only = tmp_path / "comments.yaml"
        comments_only.write_text("# just a comment\n")
        with pytest.raises(ValueError):
            load_config(["--config", str(comments_only)])

    def test_project_root_config_yaml_loads_all_fields(self):
        root_config = Path(__file__).parent.parent / "config.yaml"
        cfg = load_config(["--config", str(root_config)])
        assert cfg.volume_name == "/Volumes/odp_adw_mvp_n/migration/utilities/community_detection/"
        assert cfg.input_dependency_name == "ETL-table-dependencies_20251223_1218.csv"
        assert cfg.outofscope_stream_file_name == "out-of-scopte-streams.csv"
        assert cfg.report_dependency_file_name == "stream_to_report_mapping_new.csv"
        assert cfg.table_size_file_name == "table-space-in-gb_20251201_1352.csv"

    def test_yaml_volume_name_propagates_to_dependency_input_path(self, yaml_file):
        path = yaml_file({**self.MANDATORY, "outofscope_stream_file_name": "oos.csv"})
        cfg = load_config(["--config", path])
        assert cfg.dependency_input_path == "/yaml/vol/yaml-dep.csv"
        assert cfg.outofscope_stream_path == "/yaml/vol/oos.csv"


# ---------------------------------------------------------------------------
# load_config — priority: CLI > YAML
# ---------------------------------------------------------------------------


class TestLoadConfigPriority:
    MANDATORY_YAML = {"volume_name": "/yaml/", "input_dependency_name": "yaml-dep.csv"}

    def test_cli_overrides_yaml_volume_name(self, yaml_file):
        path = yaml_file(self.MANDATORY_YAML)
        cfg = load_config(["--config", path, "--volume-name", "/cli/"])
        assert cfg.volume_name == "/cli/"

    def test_cli_overrides_yaml_input_dependency_name(self, yaml_file):
        path = yaml_file(self.MANDATORY_YAML)
        cfg = load_config(["--config", path, "--input-dependency-name", "cli.csv"])
        assert cfg.input_dependency_name == "cli.csv"

    def test_cli_overrides_yaml_optional_field(self, yaml_file):
        path = yaml_file({**self.MANDATORY_YAML, "outofscope_stream_file_name": "yaml-oos.csv"})
        cfg = load_config(["--config", path, "--outofscope-stream-file-name", "cli-oos.csv"])
        assert cfg.outofscope_stream_file_name == "cli-oos.csv"

    def test_yaml_field_not_in_cli_is_preserved(self, yaml_file):
        path = yaml_file({**self.MANDATORY_YAML, "outofscope_stream_file_name": "yaml-oos.csv"})
        cfg = load_config(["--config", path, "--volume-name", "/cli/"])
        assert cfg.volume_name == "/cli/"                          # CLI wins
        assert cfg.outofscope_stream_file_name == "yaml-oos.csv"  # YAML preserved

    def test_mandatory_from_cli_optional_from_yaml(self, yaml_file):
        path = yaml_file({
            **self.MANDATORY_YAML,
            "report_dependency_file_name": "yaml-rep.csv",
        })
        cfg = load_config(["--config", path, "--volume-name", "/cli/"])
        assert cfg.volume_name == "/cli/"
        assert cfg.input_dependency_name == "yaml-dep.csv"
        assert cfg.report_dependency_file_name == "yaml-rep.csv"

    def test_mandatory_split_across_yaml_and_cli(self, yaml_file):
        # volume_name from YAML, input_dependency_name from CLI
        path = yaml_file({"volume_name": "/yaml/"})
        cfg = load_config(["--config", path, "--input-dependency-name", "cli.csv"])
        assert cfg.volume_name == "/yaml/"
        assert cfg.input_dependency_name == "cli.csv"

    def test_cli_overrides_both_fields_all_layers(self, yaml_file):
        path = yaml_file({**self.MANDATORY_YAML, "outofscope_stream_file_name": "yaml-oos.csv"})
        cfg = load_config([
            "--config", path,
            "--volume-name", "/cli/",
            "--input-dependency-name", "cli.csv",
            "--outofscope-stream-file-name", "cli-oos.csv",
        ])
        assert cfg.volume_name == "/cli/"
        assert cfg.input_dependency_name == "cli.csv"
        assert cfg.outofscope_stream_file_name == "cli-oos.csv"

    def test_optional_field_absent_in_yaml_and_cli_is_none(self, yaml_file):
        path = yaml_file(self.MANDATORY_YAML)
        cfg = load_config(["--config", path])
        assert cfg.table_size_file_name is None
        assert cfg.table_size_path is None


# ---------------------------------------------------------------------------
# load_config — error cases
# ---------------------------------------------------------------------------


class TestLoadConfigErrors:
    def test_nonexistent_config_file_raises_file_not_found(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.yaml")
        with pytest.raises(FileNotFoundError):
            load_config(["--config", missing])

    def test_invalid_yaml_raises_yaml_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("{invalid_yaml: [unclosed bracket")
        with pytest.raises(yaml.YAMLError):
            load_config(["--config", str(bad)])

    def test_unknown_cli_argument_raises_system_exit(self):
        with pytest.raises(SystemExit):
            load_config(["--unknown-arg", "value"])


# ---------------------------------------------------------------------------
# Integration — leiden.py variable-assignment pattern
# ---------------------------------------------------------------------------


class TestLeidenVariableAssignments:
    """
    Replicate the exact variable bindings from leiden.py lines 41-51 and
    verify they produce the correct values.  Any future divergence between
    config.py properties and leiden.py assignments will be caught here.
    """

    BASE_ARGS = [
        "--volume-name", "/test/vol/",
        "--input-dependency-name", "deps.csv",
        "--outofscope-stream-file-name", "oos.csv",
        "--report-dependency-file-name", "rep.csv",
        "--table-size-file-name", "size.csv",
    ]

    @patch("migration_planner.utils.config.datetime")
    def test_all_leiden_variables_match_config_properties(self, mock_dt):
        mock_dt.now.return_value = FIXED_DT
        cfg = load_config(self.BASE_ARGS)

        # Exact assignment block from leiden.py
        volume_path = cfg.volume_name
        dependency_input_path = cfg.dependency_input_path
        outofscope_stream_path = cfg.outofscope_stream_path
        report_dependency = cfg.report_dependency_path
        table_size = cfg.table_size_path
        output_path = cfg.output_path
        latest_path = cfg.latest_path

        assert volume_path == "/test/vol/"
        assert dependency_input_path == "/test/vol/deps.csv"
        assert outofscope_stream_path == "/test/vol/oos.csv"
        assert report_dependency == "/test/vol/rep.csv"
        assert table_size == "/test/vol/size.csv"
        assert output_path == f"/test/vol/community_detection_output_latest/{FIXED_DIR_NAME}/"
        assert latest_path == "/test/vol/community_detection_output_latest/"

    def test_output_path_is_deeper_than_latest_path(self):
        cfg = load_config(self.BASE_ARGS)
        assert cfg.output_path.startswith(cfg.latest_path)
        assert len(cfg.output_path) > len(cfg.latest_path)

    def test_optional_paths_are_none_when_not_provided(self):
        cfg = load_config(["--volume-name", "/v/", "--input-dependency-name", "dep.csv"])
        assert cfg.outofscope_stream_path is None
        assert cfg.report_dependency_path is None
        assert cfg.table_size_path is None

    @patch("migration_planner.utils.config.datetime")
    def test_leiden_variables_with_yaml_config(self, mock_dt, yaml_file):
        mock_dt.now.return_value = FIXED_DT
        path = yaml_file({
            "volume_name": "/yaml/base/",
            "input_dependency_name": "yaml-dep.csv",
            "outofscope_stream_file_name": "yaml-oos.csv",
        })
        cfg = load_config(["--config", path])

        assert cfg.dependency_input_path == "/yaml/base/yaml-dep.csv"
        assert cfg.outofscope_stream_path == "/yaml/base/yaml-oos.csv"
        assert cfg.report_dependency_path is None
        assert cfg.output_path == f"/yaml/base/community_detection_output_latest/{FIXED_DIR_NAME}/"
