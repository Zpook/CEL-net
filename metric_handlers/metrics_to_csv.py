from metric_handlers import Metric
from typing import Dict, List

from util.csv_wrapper import CSVWrapper


class MetricsToCsv:
    def __init__(self, fileDir: str, metrics: List[Metric]) -> None:

        self._fileDir = fileDir
        self._metrics = metrics
        self._rows = 0

        self._fieldNames: List[str] = []
        metric: Metric

        for metric in metrics:

            if metric.name is None:
                metric.name = "Unnamed " + self._fieldNames.__len__().__str__()

            self._fieldNames.append(metric.name)

    def AddMetrics(self, metrics: List[Metric]):
        for metric in metrics:
            self.AddOneMetric(metric)

    def AddOneMetric(self, metric: Metric):
        if metric.name is None:
            metric.name = "Unnamed " + self._metrics.__len__().__str__()

        self._fieldNames.append(metric.name)
        self._metrics.append(metric)

    def Write(self):
        dictList: List[Dict] = []

        for metric in self._metrics:
            rowInThisMetric = -1

            for rowInThisMetric in range(metric.data.__len__()):

                if dictList.__len__() <= rowInThisMetric:
                    dictList.append({})

                currDict = dictList[rowInThisMetric]

                if rowInThisMetric < metric.data.__len__():
                    currDict[metric.name] = metric.data[rowInThisMetric]
                else:
                    currDict[metric.name] = ""

        csvSaver = CSVWrapper(self._fileDir, self._fieldNames)
        csvSaver.SaveRows(dictList)
        csvSaver.Close()
