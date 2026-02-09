import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class EvidenceItem:
    source_type: str
    title: str
    url: str
    snippet: str
    iteration: int
    toc_item: str
    captured_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class EvidenceLocker:
    def __init__(self):
        self.entries: List[EvidenceItem] = []

    def add_entry(self, entry: EvidenceItem) -> None:
        self.entries.append(entry)

    def to_json(self) -> str:
        return json.dumps([asdict(entry) for entry in self.entries], indent=2)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def save_csv(self, path: str) -> None:
        if not self.entries:
            return
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=asdict(self.entries[0]).keys())
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(asdict(entry))

    def by_toc_item(self, toc_item: str) -> List[EvidenceItem]:
        return [entry for entry in self.entries if entry.toc_item == toc_item]

    def latest_snippet(self, toc_item: str) -> Optional[str]:
        matches = self.by_toc_item(toc_item)
        return matches[-1].snippet if matches else None
