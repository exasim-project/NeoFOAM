# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""``system/coupling.yaml`` read into :class:`CouplingConfig`.

``cases/coupling_config/minimal`` sets only the required entries, every other
``cases/coupling_config/<name>`` breaks one rule; a broken chain is
multiRun's ``cases/couplings/chain`` with one entry changed.
"""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from neofoam.framework.coupling import (
    CouplingConfig,
    FieldPort,
    InProcessCommChannel,
    NearestNeighborMapping,
    comm_channel_timeout,
    participant_exchanges,
)
from neofoam.solver.regions import ControlDictConfig

CASES = Path(__file__).parent / "cases"
# the cases multiRun's tests share
MULTI_RUN = Path(__file__).parents[2] / "solver" / "multiRun" / "cases"
CHAIN = MULTI_RUN / "couplings" / "chain"


def _chain() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((CHAIN / "system" / "coupling.yaml").read_text())
    return data


def test_coupling_config_reads_every_entry() -> None:
    config = CouplingConfig.load(case_dir=MULTI_RUN / "coupled_slabs")

    assert config.model_dump() == {
        "ports": {},
        "exchanges": [
            {
                "name": "interface",
                "ports": {"left": "right", "right": "left"},
                "mapping": {"type": "nearest-neighbor"},
            }
        ],
        "couplings": [
            {
                "name": "coupling",
                "scheme": "parallel",
                "maxIterations": 100,
                "exchanges": [
                    {"name": "interface", "acceleration": {"type": "aitken", "relaxation": 0.5}}
                ],
                "convergence": [
                    {"data": "T_sink", "type": "relative", "limit": 1e-12},
                    {"data": "h", "type": "relative", "limit": 1e-12},
                ],
                "wraps": {},
            }
        ],
        "commChannelTimeout": 600.0,
    }


def test_coupling_config_defaults_the_optional_entries() -> None:
    config = CouplingConfig.load(case_dir=CASES / "coupling_config" / "minimal")

    assert config.model_dump() == {
        "ports": {},
        "exchanges": [
            {
                "name": None,
                "ports": {"solid": "interface", "fluid": "interface"},
                "mapping": {"type": "nearest-neighbor"},
            }
        ],
        "couplings": [
            {
                "name": "thermal",
                "scheme": "serial",
                "maxIterations": 100,
                "exchanges": [{"name": "solid-fluid", "acceleration": None}],
                "convergence": [{"data": "T", "type": "relative", "limit": 1e-5}],
                "wraps": {},
            }
        ],
        "commChannelTimeout": 600.0,
    }


def test_coupling_config_reads_a_list_of_exchanges() -> None:
    config = CouplingConfig.load(case_dir=MULTI_RUN.parent / "limitations" / "cases" / "chain")

    assert [(e.name, e.ports, type(e.mapping)) for e in config.exchanges] == [
        (None, {"a": "to_b", "b": "to_a"}, NearestNeighborMapping),
        (None, {"b": "to_c", "c": "to_b"}, NearestNeighborMapping),
    ]


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("unknown_scheme", r"couplings.0.scheme\n.*Input should be 'serial' or 'parallel'"),
        (
            "unknown_mapping",
            r"exchanges.0.mapping.mapping\n.*Input tag 'rbf' .* expected tags: 'identity', "
            r"'nearest-neighbor'",
        ),
        (
            "unknown_acceleration",
            r"acceleration.acceleration\n.*Input tag 'anderson' .* expected tags: 'constant', "
            r"'aitken'",
        ),
        (
            "acceleration_option_typo",
            r"acceleration.aitken.relaxation\n.*Field required.*\n.*\n.*acceleration.aitken."
            r"relax\n.*Extra inputs",
        ),
        (
            "relaxation_above_one",
            r"acceleration.constant.relaxation\n.*less than or equal to 1",
        ),
        ("unknown_convergence", r"convergence.0.convergence\n.*Input tag 'absolute' .* 'relative'"),
        (
            "convergence_option_typo",
            r"convergence.relative.limit\n.*Field required.*\n.*\n.*convergence.relative."
            r"lmit\n.*Extra inputs",
        ),
        ("no_convergence", r"couplings.0.convergence\n.*at least 1 item"),
        ("three_regions", r"exchanges.0.ports\n.*at most 2 items"),
        ("top_level_convergence", r"couplings\n.*Field required.*\n.*\nconvergence\n.*Extra"),
        (
            "removed_sides_key",
            r"exchanges.0.ports\n.*Field required.*\n.*\nexchanges.0.sides\n.*Extra",
        ),
    ],
)
def test_coupling_config_rejects_an_invalid_file(case: str, error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        CouplingConfig.load(case_dir=CASES / "coupling_config" / case)


def test_coupling_config_gives_every_loop_an_acceleration_of_its_own() -> None:
    config = CouplingConfig.load(case_dir=CHAIN)

    first, second = config.loops()[0], config.loops()[0]

    assert first.acceleration["b-c"] is not second.acceleration["b-c"]


def test_a_loop_without_a_scheme_is_serial() -> None:
    data = _chain()
    del data["couplings"][1]["scheme"]

    loops = CouplingConfig.model_validate(data).loops()

    assert [(loop.name, loop.scheme) for loop in loops] == [
        ("outer", "parallel"),
        ("inner", "serial"),
    ]


def test_a_serial_loop_may_enclose_a_chained_loop() -> None:
    """outer couples b and c, inner a and b: they share b only."""
    data = _chain()
    data["couplings"][0]["scheme"] = "serial"

    loops = CouplingConfig.model_validate(data).loops()

    assert [loop.scheme for loop in loops] == ["serial", "parallel"]


def test_a_serial_loop_enclosing_a_loop_of_two_of_its_regions_raises_as_the_file_loads() -> None:
    with pytest.raises(
        ValidationError,
        match=r"coupling 'fsi' is serial but encloses 'cht', which couples \['a', 'b'\] as "
        r"well; .* make 'fsi' parallel",
    ):
        CouplingConfig.load(case_dir=MULTI_RUN / "couplings" / "serial_encloses")


def _broken(path: str, value: Any) -> dict[str, Any]:
    """The chain with the entry at ``path`` (``.``-separated keys and indices) set to ``value``."""
    data = copy.deepcopy(_chain())
    *parents, last = path.split(".")
    target: Any = data
    for key in parents:
        target = target[int(key)] if isinstance(target, list) else target[key]
    if isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value
    return data


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (_broken("couplings.1.name", "outer"), "two coupling loops are named 'outer'"),
        (
            _broken("couplings.1.exchanges", ["a-b", "b-c"]),
            "exchange 'b-c' is in the coupling loops 'outer' and 'inner'",
        ),
        (_broken("couplings.1.exchanges", ["b-c-2"]), "names the exchange 'b-c-2', which does not"),
        (
            _broken("couplings", _chain()["couplings"][:1]),
            r"the exchanges \['a-b'\] are in no coupling loop",
        ),
        (
            _broken("exchanges.1", {"name": "a-b", "ports": {"b": "to_c", "c": "to_b"}}),
            "two exchanges are named 'a-b'",
        ),
        (
            _broken("exchanges.1", {"name": "b-c", "ports": {"a": "to_b", "c": "to_b"}}),
            r"a.to_b is in the exchanges 'a-b' and 'b-c'; a coupled port is in one exchange",
        ),
        (_broken("couplings.0.maxIteration", 5), r"couplings.0.maxIteration\n.*Extra inputs"),
        (
            _broken("couplings.0.exchanges.0.acceleration", {"type": "iqn-ils", "relaxation": 0.5}),
            r"couplings.0.exchanges.0.acceleration.acceleration\n.*Input tag 'iqn-ils' .* "
            r"expected tags: 'constant', 'aitken'",
        ),
        (
            _broken("couplings.1.wraps.b.operation", ["solve_x"]),
            r"couplings.1.wraps.b.operation\n.*Extra inputs",
        ),
        (
            _broken("couplings.1.wraps.b.downstream", False),
            r"couplings.1.wraps.b.downstream\n.*Extra inputs",
        ),
        (
            _broken("couplings.1.wraps", {"c": {"operations": ["solve_x"]}}),
            r"loop 'inner' places its op in the regions \['c'\], which it does not couple",
        ),
        (_broken("maxIterations", 5), r"maxIterations\n.*Extra inputs"),
        (_broken("coupling", []), r"coupling\n.*Extra inputs"),
    ],
    ids=[
        "duplicate-loop-name",
        "exchange-in-two-loops",
        "unknown-exchange",
        "exchange-in-no-loop",
        "duplicate-exchange-name",
        "patch-in-two-exchanges",
        "typo-in-a-loop",
        "unknown-acceleration",
        "typo-in-wraps",
        "downstream-in-wraps",
        "wraps-a-region-of-another-loop",
        "top-level-maxIterations",
        "typo-at-the-top",
    ],
)
def test_coupling_config_rejects_a_broken_chain(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        CouplingConfig.model_validate(data)


@pytest.mark.parametrize("seconds", [-1, float("inf")])
def test_coupling_config_rejects_a_broken_comm_channel_timeout(seconds: float) -> None:
    with pytest.raises(ValidationError, match="commChannelTimeout"):
        CouplingConfig.model_validate({**_chain(), "commChannelTimeout": seconds})


def test_coupling_config_reads_the_field_ports() -> None:
    config = CouplingConfig.load(case_dir=MULTI_RUN / "field_ports")

    assert config.field_ports() == {"left": [FieldPort("T_cells", "T")]}
    assert [(e.name, e.ports) for e in config.exchanges] == [
        ("interface", {"left": "right", "right": "left"}),
        ("sensor", {"left": "T_cells", "meter": "T_left"}),
    ]


def test_coupling_config_checks_the_directions_of_its_field_ports() -> None:
    with pytest.raises(ValidationError, match="exchange 'sensor': T_cells is inout and T_in is"):
        CouplingConfig.load(case_dir=CASES / "coupling_config" / "field_port_directions")


def test_the_config_gives_each_region_its_coupling_as_participant_exchanges_does() -> None:
    config = CouplingConfig.load(case_dir=CHAIN)
    comm_channel = InProcessCommChannel()
    args = (config.coupling_exchanges(), comm_channel, config.loops(), config.field_ports())

    assert config.participant_exchanges(["a", "b", "c"], comm_channel) == participant_exchanges(
        ["a", "b", "c"], *args
    )


def test_the_config_s_coupling_of_regions_the_case_does_not_run_raises() -> None:
    config = CouplingConfig.load(case_dir=CHAIN)

    with pytest.raises(
        ValueError, match=r"couple the regions \['c'\], which are not in \['a', 'b'\]"
    ):
        config.participant_exchanges(["a", "b"], InProcessCommChannel())


def _timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: Any) -> Any:
    """The comm channel timeout, the chain's ``commChannelTimeout`` set to ``seconds``.

    ``"none"``: no case file; ``"default"``: the file as it is.
    """
    (tmp_path / "system").mkdir()
    if seconds != "none":
        config = CouplingConfig.load(case_dir=CHAIN)
        if seconds != "default":
            config.commChannelTimeout = seconds
        config.save(case_dir=tmp_path)
    monkeypatch.chdir(tmp_path)
    return comm_channel_timeout()


@pytest.mark.parametrize(
    ("seconds", "timeout"),
    [("none", 600.0), ("default", 600.0), (5, 5.0), (0, None), (None, None)],
    ids=["no-file", "default", "seconds", "zero", "null"],
)
def test_the_comm_channel_timeout_is_the_file_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: Any, timeout: Any
) -> None:
    assert _timeout(tmp_path, monkeypatch, seconds) == timeout


def test_region_solvers_take_a_plain_name_as_one_process() -> None:
    config = ControlDictConfig.load(case_dir=MULTI_RUN / "region_solvers" / "entries")

    assert config.solvers() == {"left": "a", "right": "b"}
    assert config.processes() == {"left": 1, "right": 3}


def test_region_solvers_default_a_region_to_one_process() -> None:
    config = ControlDictConfig.load(case_dir=MULTI_RUN / "region_solvers" / "default_processes")

    assert config.processes() == {"left": 1}


@pytest.mark.parametrize("name", ["zero_processes", "unknown_entry", "no_solver"])
def test_region_solvers_reject_an_invalid_entry(name: str) -> None:
    with pytest.raises(ValidationError):
        ControlDictConfig.load(case_dir=MULTI_RUN / "region_solvers" / name)
