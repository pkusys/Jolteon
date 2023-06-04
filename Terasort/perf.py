import re
import json

num = 128
type_tag = 1
func_type = 'map' if type_tag == 0 else 'reduce'

if __name__ == '__main__':
    billed_time = 0

    # parse_time = 0
    read_time = 0
    # record_creation_time = 0
    sort_time = 0
    write_time = 0

    avg_duration = 0
    max_duration = 0
    min_duration = 1e9

    for id in range(num):
        fn = './logs/terasort-{}-task{}.log'.format(func_type, id)
        with open(fn, 'r') as f:
            lines = f.readlines()

        # che = False
        
        # for line in lines:
        #     li = re.split('[ |\t]', line)
        #     if 'Billed' in li and 'Duration:' in li:
        #         idx = li.index('Duration:')
        #         idx = li.index('Duration:', idx + 1)
        #         billed_time += int(li[idx+1])    # ms
        #         che = True

        # if not che:
        #     assert False

        # read the last line and convert it to dict
        line = lines[-1]
        li = line.replace('\n', ' ').replace('\t', ' ').replace(':', ' ') \
            .replace(',', ' ').replace('\'', ' ').replace('\"', ' ') \
            .replace('}', ' ').replace('{', ' ').split(' ')
        li = [i for i in li if i != '']  # delete empty string
        res = dict()
        for i in range(0, len(li), 2):
            res[li[i]] = int(float(li[i+1]))

        if 'billed_duration' not in res or \
            'read_duration' not in res or 'record_creation_duration' not in res or \
            'sort_duration' not in res or 'write_duration' not in res or \
            'duration' not in res:
            assert False
        else:
            billed_time += res['billed_duration']
            # parse_time += res['parse_duration']
            read_time += res['read_duration'] + res['record_creation_duration']
            sort_time += res['sort_duration']
            write_time += res['write_duration']
            avg_duration += res['duration']
            max_duration = max(max_duration, res['duration'])
            min_duration = min(min_duration, res['duration'])
        
    if func_type == 'map':
        print("Billing: {} $".format(billed_time * 1792 / 1024 * 0.0000000167 + num * 0.2 / 1000000))
    else:
        print("Billing: {} $".format(billed_time * 3584 / 1024 * 0.0000000167 + num * 0.2 / 1000000))
    print("Avg Duration: {} ms".format(avg_duration / num))
    print("Max Duration: {} ms".format(max_duration))
    print("Min Duration: {} ms".format(min_duration))
    print("Avg Read: {} ms".format(read_time / num))
    print("Avg Sort: {} ms".format(sort_time / num))
    print("Avg Write: {} ms".format(write_time / num))