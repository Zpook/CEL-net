import csv
from typing import Any, Dict, Iterable


class CSVWrapper:
    def __init__(self, fileDir: str, fields: Iterable[str]) -> None:
        self._file = open(fileDir, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader()

    def SaveRow(self, dataDict: Dict[str, Any]):
        self._writer.writerow(dataDict)

    def SaveRows(self, data: Iterable[Dict[str, Any]]):
        self._writer.writerows(data)

    def Close(self):
        self._file.close()
