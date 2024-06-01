import json


class Violation:
    def __init__(self, type, start, end, video_name):
        self.type = type
        self.start = start
        self.end = end
        minutes, seconds = map(int, start.split(':'))
        total_seconds = minutes * 60 + seconds
        self.start_seconds = total_seconds
        minutes, seconds = map(int, end.split(':'))
        total_seconds = minutes * 60 + seconds
        self.end_seconds = total_seconds
        self.video_name = video_name

def get_uniq_video_names(violation_dict, type):
    uniq_video_names = set()
    for violation in violation_dict[type]:
        uniq_video_names.add(violation.video_name)
    return uniq_video_names

def get_video_violations(violation_dict, type, video_name):
    violations = violation_dict[type]
    violations = filter(lambda violation: violation.video_name == video_name, violations)
    violations = list(violations)
    return violations

def get_violation_dict_modified(violation_dict):
    modified_dict = {}
    for violation_type in violation_dict.keys():
        modified_dict[violation_type] = {}
        uniq_video_names = get_uniq_video_names(violation_dict, violation_type)
        for video_name in uniq_video_names:
            modified_dict[violation_type][video_name] = get_video_violations(violation_dict, violation_type, video_name)
    return modified_dict

def get_violations_dict(violations_json):
    violations_dict = {}
    #with open('violations.json', encoding='utf-8') as f:
        #violations = json.load(f)
    violations = violations_json
    for v in violations:
        if v['type'] not in violations_dict.keys():
            violations_dict[v['type']] = []
        violations_dict[v['type']].append(Violation(v['type'], v['start'], v['end'], v['name'].replace(" ", "_")))

    return violations_dict