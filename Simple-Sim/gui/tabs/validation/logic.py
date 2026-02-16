"""Logic for the Validation Tab."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
from datetime import datetime

from gui.utils.validation_suite import run_validation_suite
from gui.utils.flag_manager import FlagManager


class ValidationLogic:
    def __init__(self, sim_root: Path) -> None:
        self.sim_root = sim_root
        self.flag_manager: FlagManager | None = None

    def set_dataset(self, dataset_dir: Path) -> None:
        self.flag_manager = FlagManager(dataset_dir)

    def get_datasets(self) -> List[Path]:
        sim_data = self.sim_root / "outputs" / "sim_data"
        runs = sim_data / "runs"
        versions = sim_data / "versions"
        runs.mkdir(parents=True, exist_ok=True)
        versions.mkdir(parents=True, exist_ok=True)
        
        cand: list[Path] = []
        cand.extend([p for p in runs.iterdir() if p.is_dir()])
        cand.extend([p for p in versions.glob("*/*") if p.is_dir()])
        return cand

    def run_validation(self, dataset_dir: Path, checks: Dict[str, bool]) -> Dict[str, Any]:
        results = run_validation_suite(
            dataset_dir,
            run_schema=checks.get("schema", False),
            run_outliers=checks.get("outliers", False),
            run_duplicates=checks.get("duplicates", False),
            run_edge_cases=checks.get("edge_cases", False),
        )

        if self.flag_manager:
            for outlier in results.get('outliers', []):
                self.flag_manager.add_flag(outlier['sample_id'], outlier['flag'], outlier['reason'])
            for edge_case in results.get('edge_cases', []):
                self.flag_manager.add_flag(edge_case['sample_id'], edge_case['flag'], edge_case['reason'])
            for id1, id2, distance in results.get('duplicates', []):
                reason = f'Duplicate of {id2} (distance={distance})'
                self.flag_manager.add_flag(id1, 'DUPLICATE', reason)

        return results

    def get_all_flagged_samples(self) -> List[str]:
        if not self.flag_manager:
            return []
        return self.flag_manager.get_all_flagged_samples()

    def get_flags(self, sample_id: str) -> List[Dict[str, str]]:
        if not self.flag_manager:
            return []
        return self.flag_manager.get_flags(sample_id)

    def remove_flag(self, sample_id: str, flag_type: str) -> None:
        if self.flag_manager:
            self.flag_manager.remove_flag(sample_id, flag_type)

    def clear_all_flags(self) -> None:
        if self.flag_manager:
            for sample_id in self.flag_manager.get_all_flagged_samples():
                self.flag_manager.remove_flag(sample_id)

    def export_report(self, results: Dict[str, Any], dataset_dir: Path) -> str:
        output_path = dataset_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            import numpy as _np
        except ImportError:
            _np = None

        def _jsonify(obj):
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, Path):
                return str(obj)
            if _np is not None:
                if isinstance(obj, _np.generic):
                    return obj.item()
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
            if isinstance(obj, dict):
                return {str(k): _jsonify(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_jsonify(v) for v in obj]
            return str(obj)

        with open(output_path, 'w') as f:
            json.dump(_jsonify(results), f, indent=2)
        
        return str(output_path)
