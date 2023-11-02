import re
from dataclasses import dataclass
from os import path
from typing import Union


@dataclass
class DatasetInfo:
    set_name: str = None
    info: dict = None

    def to_dict(self):
        return self.__dict__

    def to_ntu_filename(self):
        # Ntu would have "set", "camera", "person", "replication", "action"
        template = "S{:03d}C{:03d}P{:03d}R{:03d}A{:03d}"
        info_list = [self.info.get(key, 0) for key in ["set", "camera", "person", "replication", "action"]]
        return template.format(*info_list)

    def to_ut_filename(self):
        template = "{}_{}_{}"
        info_list = [self.info.get(key, 0) for key in ["subject", "camera", "action"]]
        return template.format(*info_list)

    def to_filename(self):
        if self.set_name == "ntu":
            return self.to_ntu_filename()
        elif self.set_name == "ut":
            return self.to_ut_filename()
        else:
            raise ValueError(f"Not supported set_name {self.set_name}")


def name_to_ntu_data(filepath: str) -> Union[DatasetInfo, None]:
    filename = path.split(filepath)[-1]
    match = ntu_name_template.match(filename)
    if match:
        data = match.groupdict()
        data = {k: int(v) for k, v in data.items()}
        return DatasetInfo("ntu", data)
    return None


def name_to_ut_data(filepath: str) -> Union[DatasetInfo, None]:
    filename = path.split(filepath)[-1]
    match = ut_name_template.match(filename)
    if match:
        data = match.groupdict()
        data = {k: int(v) for k, v in data.items()}
        return DatasetInfo("ut", data)
    return None


ut_name_template = re.compile("(?P<subject>[0-9]+)_(?P<camera>[0-9]+)_(?P<action>[0-9]+)\.(avi|apskel\.pkl|skeleton)")

ntu_name_template = re.compile(
    "[A-Z](?P<set>[0-9]+)[A-Z](?P<camera>[0-9]+)[A-Z](?P<person>[0-9]+)[A-Z](?P<replication>[0-9]+)[A-Z](?P<action>[0-9]+)(?:_rgb)?\.(?:avi|.+\.apskel.pkl|skeleton)"
)

name_info_func_map = {
    "ntu": name_to_ntu_data,
    "ut": name_to_ut_data,
}
