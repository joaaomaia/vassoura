from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, Dict, Type, Any
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px

from .audit import AuditTrail

__all__ = ["ReportManager", "register_section", "SECTION_REGISTRY"]


class ReportSection(Protocol):
    name: str

    def generate(self) -> str: ...


SECTION_REGISTRY: Dict[str, Type[Any]] = {}


def register_section(name: str):
    def decorator(cls: Type[Any]):
        SECTION_REGISTRY[name] = cls
        return cls

    return decorator


@dataclass
class OverviewSection:
    audit: AuditTrail
    snapshot_names: list[str]
    dataset_shape: tuple[int, int]

    name: str = "overview"

    def generate(self) -> str:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshots = ", ".join(self.snapshot_names)
        html = "<h2>Overview</h2>"
        html += f"<p>Date: {date}</p>"
        html += f"<p>Dataset shape: {self.dataset_shape}</p>"
        html += f"<p>Snapshots: {snapshots}</p>"
        return html


@dataclass
class PerformanceSection:
    metrics: pd.DataFrame

    name: str = "performance"

    def generate(self) -> str:
        html = "<h2>Performance</h2>" + self.metrics.to_html()
        return html


@dataclass
class FeatureImportanceSection:
    importance: pd.Series

    name: str = "feature_importance"

    def generate(self) -> str:
        imp = self.importance.sort_values(ascending=False).head(20)
        fig = px.bar(imp, orientation="h")
        return "<h2>Feature Importance</h2>" + fig.to_html(include_plotlyjs="cdn")


@dataclass
class AuditDiffSection:
    audit: AuditTrail
    base: str
    new: str

    name: str = "audit_diff"

    def generate(self) -> str:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.audit.compare_snapshots(self.base, self.new)
        content = buf.getvalue().replace("\n", "<br>")
        return f"<h2>Audit Diff</h2><pre>{content}</pre>"


@register_section("overview")
class RegisteredOverview(OverviewSection):
    pass


@register_section("performance")
class RegisteredPerformance(PerformanceSection):
    pass


@register_section("feature_importance")
class RegisteredFeatureImportance(FeatureImportanceSection):
    pass


@register_section("audit_diff")
class RegisteredAuditDiff(AuditDiffSection):
    pass


class ReportManager:
    def __init__(
        self,
        mandatory_sections: tuple[str, ...] = ("overview", "performance"),
        async_render: bool = False,
        theme: str = "flatly",
    ) -> None:
        self.mandatory_sections = list(mandatory_sections)
        self.async_render = async_render
        self.theme = theme
        self.sections: Dict[str, ReportSection] = {}

    def add_section(self, section: ReportSection) -> None:
        self.sections[section.name] = section

    def _render_section(self, section: ReportSection) -> str:
        return section.generate()

    def render(self, path: str | Path = "reports/report.html") -> Path:
        order = self.mandatory_sections + [
            n for n in self.sections.keys() if n not in self.mandatory_sections
        ]
        sections = [self.sections[n] for n in order if n in self.sections]

        if self.async_render:

            async def _collect() -> list[str]:
                return await asyncio.gather(
                    *[asyncio.to_thread(self._render_section, s) for s in sections]
                )

            html_parts: list[str] = asyncio.run(_collect())
        else:
            html_parts = [self._render_section(s) for s in sections]

        toc = (
            "<ul>"
            + "".join(
                f"<li><a href='#{n}'>{n.title()}</a></li>"
                for n in order
                if n in self.sections
            )
            + "</ul>"
        )

        body = "".join(
            f"<section id='{sec.name}'>{html}</section>"
            for sec, html in zip(sections, html_parts)
        )
        css = f"https://cdn.jsdelivr.net/npm/bootswatch@5.3.0/dist/{self.theme}/bootstrap.min.css"
        final_html = f"<html><head><link rel='stylesheet' href='{css}'></head><body>{toc}{body}</body></html>"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(final_html, encoding="utf-8")
        return path
