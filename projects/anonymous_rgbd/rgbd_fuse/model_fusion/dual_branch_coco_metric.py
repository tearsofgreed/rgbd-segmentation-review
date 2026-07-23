from copy import deepcopy
from typing import List, Optional

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class DualBranchCocoMetric(CocoMetric):
    """同一次评估中，同时统计 stage1 / stage2 两套实例结果。"""

    def __init__(self,
                 *args,
                 stage1_field: str = 'pred_instances_stage1',
                 stage2_field: str = 'pred_instances_stage2',
                 stage1_prefix: str = 'stage1',
                 stage2_prefix: str = 'stage2',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.stage1_field = stage1_field
        self.stage2_field = stage2_field
        self.stage1_prefix = stage1_prefix
        self.stage2_prefix = stage2_prefix

        self._stage1_results = []
        self._stage2_results = []

    def _has_field(self, data_sample, field_name: str) -> bool:
        if isinstance(data_sample, dict):
            return field_name in data_sample
        return hasattr(data_sample, field_name)

    def _get_field(self, data_sample, field_name: str):
        if isinstance(data_sample, dict):
            return data_sample[field_name]
        return getattr(data_sample, field_name)

    def _set_field(self, data_sample, field_name: str, value):
        if isinstance(data_sample, dict):
            data_sample[field_name] = value
        else:
            setattr(data_sample, field_name, value)

    def _clone_with_branch_pred(self,
                                data_samples: List,
                                branch_field: str,
                                fallback_field: Optional[str] = None) -> Optional[List]:
        branch_samples = []
        any_valid = False

        for ds in data_samples:
            new_ds = deepcopy(ds)

            if self._has_field(ds, branch_field):
                pred_instances = self._get_field(ds, branch_field)
                self._set_field(new_ds, 'pred_instances', deepcopy(pred_instances))
                any_valid = True
            elif fallback_field is not None and self._has_field(ds, fallback_field):
                pred_instances = self._get_field(ds, fallback_field)
                self._set_field(new_ds, 'pred_instances', deepcopy(pred_instances))
                any_valid = True
            else:
                return None

            branch_samples.append(new_ds)

        return branch_samples if any_valid else None

    def _collect_branch_results(self, data_batch, data_samples: List):
        old_results = self.results
        self.results = []
        super().process(data_batch, data_samples)
        collected = self.results
        self.results = old_results
        return collected

    def process(self, data_batch, data_samples: List) -> None:
        stage1_samples = self._clone_with_branch_pred(
            data_samples,
            branch_field=self.stage1_field,
            fallback_field='pred_instances'
        )
        if stage1_samples is not None:
            self._stage1_results.extend(self._collect_branch_results(data_batch, stage1_samples))

        stage2_samples = self._clone_with_branch_pred(
            data_samples,
            branch_field=self.stage2_field,
            fallback_field=None
        )
        if stage2_samples is not None:
            self._stage2_results.extend(self._collect_branch_results(data_batch, stage2_samples))

        self.results = []

    def compute_metrics(self, results: list) -> dict:
        metrics = {}

        if len(self._stage1_results) > 0:
            stage1_metrics = super().compute_metrics(self._stage1_results)
            metrics.update({f'{self.stage1_prefix}/{k}': v for k, v in stage1_metrics.items()})

        if len(self._stage2_results) > 0:
            stage2_metrics = super().compute_metrics(self._stage2_results)
            metrics.update({f'{self.stage2_prefix}/{k}': v for k, v in stage2_metrics.items()})

        self._stage1_results = []
        self._stage2_results = []

        return metrics