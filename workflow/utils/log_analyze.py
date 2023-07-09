import re

def extract_info_from_log(log):
    assert isinstance(log, str)
    
    billed_duration = 0
    duration = 0
    memory_size = 0
    memory_used = 0
    
    billed_duration = re.search(r"Billed Duration: (\d+)", log).group(1)
    duration = re.search(r"Duration: (\d+.\d+)", log).group(1)
    memory_used = re.search(r"Max Memory Used: (\d+)", log).group(1)
    memory_size = re.search(r"Memory Size: (\d+)", log).group(1)
    
    info = {
        "billed_duration": float(billed_duration),
        "duration": float(duration),
        "memory_size": float(memory_size),
        "memory_used": float(memory_used)
    }
    
    bill = caculate_bill(info)
    info['bill'] = bill
    
    return info
    
def caculate_bill(info):
    assert isinstance(info, dict)
    bill = info['billed_duration'] * info['memory_size'] / 1024 * 0.0000000167 + 0.2 / 1000000
    return bill