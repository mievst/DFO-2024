import json
import os
from openpyxl import Workbook

from services import violation_service


def export_violations_to_xlsx():
    path = os.path.join("timecodes", "timecodes.json")
    correct_path = path.replace(os.path.sep, '/')
    if os.path.exists(correct_path):
        with open(correct_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        violations_dict = violation_service.get_violations_dict(data)
        violations_dict = violation_service.get_violation_dict_modified(violations_dict)
        mDict = violations_dict

        wb = Workbook()
        ws = wb.active

        header = rf"violation_type;video_name;violation_start;violation_end;violation_start_seconds;violation_end_seconds"
        ws.append(header.split(";"))
        for k in mDict.keys():
            for video in mDict[k]:
                for violation in mDict[k][video]:
                    csv_string = rf'{k};{video};{violation.start};{violation.end};{violation.start_seconds};{violation.end_seconds}'
                    ws.append(csv_string.split(";"))
                    print(violation.start, violation.end)

    return wb,ws