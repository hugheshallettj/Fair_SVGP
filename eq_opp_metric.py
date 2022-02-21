from aif360.sklearn.metrics import equal_opportunity_difference
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from fairlearn.metrics import true_positive_rate

def eq_op_diff(y_true, y_pred):
    dataset = StandardDataset(y_true, label_name="Income Binary", favorable_classes=[1], protected_attribute_names=["sex"], privileged_classes=[[1]])
    class_metric = ClassificationMetric(
        dataset,
        y_pred,
        privileged_groups = [{'sex': 1}],
        unprivileged_groups = [{'sex': 0}]
        )
    return class_metric.equal_opportunity_difference()
