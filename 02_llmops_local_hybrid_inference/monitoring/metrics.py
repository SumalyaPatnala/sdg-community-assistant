from collections import defaultdict


class MetricsStore:
    def __init__(self):
        self.records = []

    def add(self, record: dict):
        self.records.append(record)

    def summary(self):
        if not self.records:
            return {
                "total_requests": 0,
                "success_rate": None,
                "avg_latency_ms": None,
                "by_task": {}
            }

        total = len(self.records)
        success = sum(1 for r in self.records if r.get("status") == "success")
        avg_latency = sum(r.get("latency_ms", 0) for r in self.records) / total

        by_task = defaultdict(list)
        for r in self.records:
            by_task[r.get("task", "unknown")].append(r)

        task_summary = {}
        for task, rows in by_task.items():
            task_summary[task] = {
                "count": len(rows),
                "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in rows) / len(rows), 2),
                "success_count": sum(1 for r in rows if r.get("status") == "success")
            }

        return {
            "total_requests": total,
            "success_rate": round(success / total, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "by_task": task_summary
        }
