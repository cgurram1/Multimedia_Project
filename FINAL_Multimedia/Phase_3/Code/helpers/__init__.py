import json
import platform
import re
import os
a=platform.system()

def filter_fn(a):
    if(a in ['']):
        return False
    else:
        return True

def os_path_separator():
    if(a=='Windows'):
        return "\\"
    else:
        return '/'
def format_os_string(str1):
    split_arr=re.split(r"//|/|\\|\\\\",str1)
    split=list(filter(filter_fn,split_arr))
    double_dot_idx=[idx for idx,ele in enumerate(split) if(ele in [".."]) ]
    idx_to_remove=[i-1 for i in double_dot_idx]
    idx_to_remove=idx_to_remove+double_dot_idx
    split=[ ele for idx,ele in enumerate(split) if(idx not in idx_to_remove)]
    if(a=='Windows'):
        return "\\".join(split)
    return "/".join(split)

f=open(format_os_string("D:\Project Multimedia\Phase 3\code\helpers\constants.json"),'r')
constants=json.loads(f.read())
f.close()
# print(constants)

def fetch_constant(key):
    if(constants[key]['isFilePath']):
        return format_os_string(constants[key]['value'])
    else:
        return constants[key]['value']



