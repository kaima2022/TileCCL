"""Authentic timeline event recorder and simple lane renderer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TimelineEvent:
    """One real runtime or benchmark event to be rendered on a lane."""

    name: str
    lane: str
    start_ms: float
    end_ms: float
    category: str = "span"
    rank: int | None = None
    label: str | None = None
    source: str = "runtime"
    tile_range: Mapping[str, Any] | None = None
    color: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        start_ms = float(self.start_ms)
        end_ms = float(self.end_ms)
        if not math.isfinite(start_ms) or not math.isfinite(end_ms):
            raise ValueError("TimelineEvent times must be finite")
        if end_ms < start_ms:
            raise ValueError(
                f"TimelineEvent {self.name!r} has end_ms < start_ms: "
                f"{end_ms} < {start_ms}"
            )
        if not self.source:
            raise ValueError("TimelineEvent source must not be empty")
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "lane": self.lane,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "category": self.category,
            "source": self.source,
        }
        if self.rank is not None:
            payload["rank"] = self.rank
        if self.label is not None:
            payload["label"] = self.label
        if self.tile_range is not None:
            payload["tile_range"] = dict(self.tile_range)
        if self.color is not None:
            payload["color"] = self.color
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TimelineEvent":
        return cls(
            name=str(payload["name"]),
            lane=str(payload["lane"]),
            start_ms=float(payload["start_ms"]),
            end_ms=float(payload["end_ms"]),
            category=str(payload.get("category", "span")),
            rank=payload.get("rank"),
            label=payload.get("label"),
            source=str(payload.get("source", "runtime")),
            tile_range=payload.get("tile_range"),
            color=payload.get("color"),
            metadata=payload.get("metadata", {}),
        )


class TimelineRecorder:
    """Records lane events and writes JSON/PNG proof artifacts."""

    def __init__(
        self,
        *,
        benchmark: str,
        path_name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.path_name = path_name
        self.metadata = dict(metadata or {})
        self._events: list[TimelineEvent] = []

    @property
    def events(self) -> tuple[TimelineEvent, ...]:
        return tuple(self._events)

    def record_span(
        self,
        *,
        name: str,
        lane: str,
        start_ms: float,
        end_ms: float,
        category: str = "span",
        rank: int | None = None,
        label: str | None = None,
        source: str = "runtime",
        tile_range: Mapping[str, Any] | None = None,
        color: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TimelineEvent:
        event = TimelineEvent(
            name=name,
            lane=lane,
            start_ms=start_ms,
            end_ms=end_ms,
            category=category,
            rank=rank,
            label=label,
            source=source,
            tile_range=tile_range,
            color=color,
            metadata=metadata or {},
        )
        self._events.append(event)
        return event

    def record_marker(
        self,
        *,
        name: str,
        lane: str,
        at_ms: float,
        category: str = "marker",
        rank: int | None = None,
        label: str | None = None,
        source: str = "runtime",
        tile_range: Mapping[str, Any] | None = None,
        color: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TimelineEvent:
        return self.record_span(
            name=name,
            lane=lane,
            start_ms=at_ms,
            end_ms=at_ms,
            category=category,
            rank=rank,
            label=label,
            source=source,
            tile_range=tile_range,
            color=color,
            metadata=metadata,
        )

    def to_payload(self) -> dict[str, Any]:
        lane_order: list[str] = []
        for event in self._events:
            if event.lane not in lane_order:
                lane_order.append(event.lane)
        return {
            "schema": "tileccl.timeline.v1",
            "benchmark": self.benchmark,
            "path_name": self.path_name,
            "metadata": self.metadata,
            "lane_order": lane_order,
            "events": [event.to_json() for event in self._events],
            "authenticity_checks": {
                "event_count": len(self._events),
                "all_events_have_source": all(bool(event.source) for event in self._events),
                "all_events_have_nonnegative_duration": all(
                    event.end_ms >= event.start_ms for event in self._events
                ),
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TimelineRecorder":
        recorder = cls(
            benchmark=str(payload["benchmark"]),
            path_name=str(payload["path_name"]),
            metadata=payload.get("metadata", {}),
        )
        for event_payload in payload.get("events", []):
            recorder._events.append(TimelineEvent.from_json(event_payload))
        return recorder

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), indent=2), encoding="utf-8")

    def render_png(
        self,
        path: Path,
        *,
        title: str | None = None,
        lane_order: Sequence[str] | None = None,
    ) -> None:
        render_timeline_png(
            self._events,
            path,
            title=title or f"{self.benchmark}: {self.path_name}",
            lane_order=lane_order,
        )


def _ordered_lanes(
    events: Iterable[TimelineEvent],
    lane_order: Sequence[str] | None,
) -> list[str]:
    if lane_order is not None:
        return list(lane_order)
    lanes: list[str] = []
    for event in events:
        if event.lane not in lanes:
            lanes.append(event.lane)
    return lanes


def render_timeline_png(
    events: Sequence[TimelineEvent],
    path: Path,
    *,
    title: str,
    lane_order: Sequence[str] | None = None,
) -> None:
    """Render a compact lane diagram from timeline events."""

    if not events:
        raise ValueError("Cannot render an empty timeline")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lanes = _ordered_lanes(events, lane_order)
    lane_to_y = {lane: len(lanes) - idx for idx, lane in enumerate(lanes)}
    max_time = max(event.end_ms for event in events)

    fig_height = max(2.8, 0.55 * len(lanes) + 1.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    default_colors = {
        "producer": "#4c78a8",
        "ingress": "#b279a2",
        "frontier": "#f58518",
        "compute": "#54a24b",
        "aggregation": "#e45756",
        "marker": "#333333",
        "span": "#72b7b2",
    }

    for lane in lanes:
        y = lane_to_y[lane]
        ax.axhline(y, color="#e8edf4", linewidth=8, alpha=0.75, zorder=0)

    for event in events:
        y = lane_to_y[event.lane]
        color = event.color or default_colors.get(event.category, "#72b7b2")
        if event.duration_ms == 0:
            ax.vlines(
                event.start_ms,
                y - 0.23,
                y + 0.23,
                color=color,
                linewidth=1.5,
                zorder=3,
            )
            if event.label:
                ax.text(
                    event.start_ms,
                    y + 0.28,
                    event.label,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    color=color,
                )
            continue
        ax.barh(
            y,
            event.duration_ms,
            left=event.start_ms,
            height=0.36,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )
        label = event.label or event.name
        if event.duration_ms >= max(0.08, max_time * 0.035):
            ax.text(
                event.start_ms + event.duration_ms / 2.0,
                y,
                label,
                fontsize=8.5,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_yticks([lane_to_y[lane] for lane in lanes])
    ax.set_yticklabels(lanes, fontsize=10, fontweight="bold")
    ax.set_xlabel("Elapsed time (ms)")
    ax.set_xlim(0, max_time * 1.06 if max_time > 0 else 1.0)
    ax.grid(axis="x", color="#d7dee8", linewidth=0.8, alpha=0.8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
