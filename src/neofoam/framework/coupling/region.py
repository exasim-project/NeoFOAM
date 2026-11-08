# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""What one region knows of the coupling of a case: :class:`RegionCoupling`.

A coupled case lists its exchanges (:class:`Link`, from
``system/regionCoupling.yaml``) and its regions in stepping order (``regionSolvers``).
:func:`region_couplings` turns them into one :class:`RegionCoupling` per coupled
region, plain data plus the region's transport, handed to the region's runtime
(``instantiate(region=..., coupling=...)``); a solver that supports coupling
keeps it in the region's coupling model (``ctx.models["coupling"]``, a
:class:`~neofoam.framework.coupling.Coupling`).

Regions that exchanges connect, directly or over other regions, form a group;
the group's last region in stepping order leads it.

Example::

    links = [Link("interface", (("left", "right"), ("right", "left")))]
    couplings = region_couplings(["left", "right"], links, InProcessTransport())
    couplings["right"].is_leader    # True
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence, TypeVar

from .loop import CouplingController, GroupSide
from .transport import Transport

T = TypeVar("T")
# The end of an exchange: (region, patch).
End = tuple[str, str]


class ControllerFactory(Protocol):
    """Builds the leader's controller from the group's sides, the leader's label and the scheme.

    ``parallel`` is the coupling's (:attr:`RegionCoupling.parallel`); the
    controller decides with it which sides receive fresh or carried data.
    """

    def __call__(
        self, sides: Sequence[GroupSide], label: str, /, *, parallel: bool
    ) -> CouplingController: ...


@dataclass(frozen=True)
class Link:
    """One exchange of a case: its name, its two ends ``(region, patch)`` and its mapping kind.

    ``mapping`` is a kind of :data:`~neofoam.framework.coupling.MAPPINGS`.
    """

    name: str
    ends: tuple[End, End]
    mapping: str = "identity"

    def partner(self, region: str) -> End:
        """The end of the exchange that is not in ``region``."""
        first, second = self.ends
        return second if first[0] == region else first


@dataclass(frozen=True)
class RegionCoupling:
    """What region ``region`` knows of the coupling of its case.

    ``group`` holds the regions coupled with it, itself included, in stepping
    order; ``links`` the exchanges of the group. ``parallel`` says whether a side
    always receives what its partner sent in the pass before (else serial: from a
    region that steps earlier, what it sent in the same pass). The leader builds
    its controller with ``controller`` (convergence, acceleration, max_iterations),
    handing it ``parallel``.
    """

    region: str
    group: tuple[str, ...]
    links: tuple[Link, ...]
    transport: Transport
    parallel: bool = False
    controller: ControllerFactory = CouplingController

    @property
    def leader(self) -> str:
        """The region of the group that decides: the last in stepping order."""
        return self.group[-1]

    @property
    def is_leader(self) -> bool:
        """Whether this region leads its group."""
        return self.region == self.leader

    @property
    def patches(self) -> tuple[str, ...]:
        """The patches of this region that an exchange connects."""
        return tuple(
            dict(link.ends)[self.region] for link in self.links if self.region in dict(link.ends)
        )


def region_couplings(
    regions: Sequence[str],
    links: Sequence[Link],
    transport: Transport,
    parallel: bool = False,
    controller: ControllerFactory = CouplingController,
) -> dict[str, RegionCoupling]:
    """The coupling of every region ``links`` connect, by region; ``regions`` in stepping order.

    Raises ``ValueError`` when a link names a region not in ``regions``, joins
    two patches of one region, or shares a patch with another link.
    """
    named = sorted({region for link in links for region, _patch in link.ends} - set(regions))
    if named:
        raise ValueError(
            f"the exchanges couple the regions {named}, which are not in {list(regions)}"
        )
    owner: dict[End, str] = {}
    for link in links:
        first, second = link.ends
        if first[0] == second[0]:
            raise ValueError(f"exchange '{link.name}' joins two patches of region {first[0]}")
        for region, patch in link.ends:
            if (region, patch) in owner:
                raise ValueError(
                    f"{region}.{patch} is in the exchanges '{owner[(region, patch)]}' and "
                    f"'{link.name}'; a coupled patch is in one exchange"
                )
            owner[(region, patch)] = link.name
    couplings: dict[str, RegionCoupling] = {}
    for region in regions:
        if region in couplings:
            continue
        group = _group(region, regions, links)
        if len(group) < 2:
            continue
        group_links = tuple(link for link in links if link.ends[0][0] in group)
        for member in group:
            couplings[member] = RegionCoupling(
                member, group, group_links, transport, parallel, controller
            )
    return couplings


def _group(region: str, regions: Sequence[str], links: Sequence[Link]) -> tuple[str, ...]:
    """The regions connected with ``region`` over ``links``, itself included, in stepping order."""
    group, todo = {region}, [region]
    while todo:
        current = todo.pop()
        for link in links:
            ends = {name for name, _patch in link.ends}
            if current in ends:
                todo += sorted(ends - group)
                group |= ends
    return tuple(name for name in regions if name in group)


def run_regions(tasks: Mapping[str, Callable[[], T]], transport: Transport) -> dict[str, T]:
    """Run every region's task on its own thread; return each task's result, by region.

    The threads report to the execution observer of the calling thread. A region
    whose task fails closes the transport for every region, so none waits for it
    forever; a region that ends closes it for itself. The first error is raised
    once every thread ended.
    """
    results: dict[str, T] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run(region: str, task: Callable[[], T]) -> None:
        try:
            results[region] = task()
        except BaseException as error:  # re-raised in the calling thread
            with lock:
                errors.append(error)
            for other in tasks:
                transport.close(other)
        finally:
            transport.close(region)

    threads = [
        threading.Thread(
            target=contextvars.copy_context().run, args=(run, region, task), name=region
        )
        for region, task in tasks.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return {region: results[region] for region in tasks}
