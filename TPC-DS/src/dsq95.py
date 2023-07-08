import pandas as pd
import numpy as np
from utils import *

'''
with ws_wh as (
  select
    ws1.ws_order_number,
    ws1.ws_warehouse_sk wh1,
    ws2.ws_warehouse_sk wh2
  from
    web_sales ws1,
    web_sales ws2
  where
    ws1.ws_order_number = ws2.ws_order_number
    and ws1.ws_warehouse_sk <> ws2.ws_warehouse_sk
)
select
  count(distinct ws_order_number) as "order count",
  sum(ws_ext_ship_cost) as "total shipping cost",
  sum(ws_net_profit) as "total net profit"
from
  web_sales ws1,
  date_dim,
  customer_address,
  web_site
where
  d_date between '1999-4-01'
  and (cast('1999-4-01' as date) + 60 days)
  and ws1.ws_ship_date_sk = d_date_sk
  and ws1.ws_ship_addr_sk = ca_address_sk
  and ca_state = 'IA'
  and ws1.ws_web_site_sk = web_site_sk
  and web_company_name = 'pri'
  and ws1.ws_order_number in (
    select
      ws_order_number
    from
      ws_wh
  )
  and ws1.ws_order_number in (
    select
      wr_order_number
    from
      web_returns,
      ws_wh
    where
      wr_order_number = ws_wh.ws_order_number
  )
order by
  count(distinct ws_order_number)
limit
  100;
'''

q95_intermediate_schema = {
    'stage0': {
        'ws_order_number': 'int64',
        'unique_count_flag': 'int64',
        'unique_value': 'int64'
    },
    'stage1': {
        'ws_order_number': 'int64'
    },
    'stage2': {
        'ws_order_number': 'int64',
        'ws_ext_ship_cost': 'float32',
        'ws_net_profit': 'float32',
        'ws_ship_date_sk': 'int64',
        'ws_ship_addr_sk': 'int64',
        'ws_web_site_sk': 'int64',
        'ws_warehouse_sk': 'int64'
    },
    'stage3': {
        'wr_order_number': 'int64'
    },
    'stage4': {
        'ws_order_number': 'int64',
        'ws_ext_ship_cost': 'float32',
        'ws_net_profit': 'float32',
        'ws_ship_date_sk': 'int64',
        'ws_ship_addr_sk': 'int64',
        'ws_web_site_sk': 'int64',
        'ws_warehouse_sk': 'int64'
    },
    'stage5': {
        'ca_address_sk': 'int64'
    },
    'stage6': {
        'ws_order_number': 'int64',
        'ws_ext_ship_cost': 'float32',
        'ws_net_profit': 'float32',
    },
    'stage7': {
        'unnique_order_number_count': 'int64',
        'ship_cost_sum': 'float32',
        'net_profit_sum': 'float32'
    }
}


def q95_stage0(key):
    [tr, tc, tw] = [0] * 3

    # Read web_sales
    t0 = time.time()
    ws = read_data(key)
    t1 = time.time()
    tr += t1 - t0

    # Select columns from web_sales
    t0 = time.time()
    wanted_columns = ['ws_order_number', 'ws_warehouse_sk']
    ws_s = ws[wanted_columns]
    # print('\n\n')
    # print(ws_s.head())
    # print('\n\n')

    # x = ws_s[ws_s['ws_order_number'] == 0]
    # print(x)
    # print("len input = {}".format(len(ws_s)))
    '''
    Group by order_number and list the unique warehouse_sk for each order_number
    Note: the result is computed from a single parallel task, but obtaining the correct result 
    needs global information, so stage 1 needs to gather the results of all stage 0's tasks
    '''
    wh_uniques = ws_s.groupby(['ws_order_number'])['ws_warehouse_sk']
    wh_unique_counts = wh_uniques.nunique()

    def get_one_unique_value(x):
        return x.unique()[0]
    
    wh_uniques = wh_uniques.apply(get_one_unique_value)
    wh_uc = pd.DataFrame({
        'ws_order_number': wh_unique_counts.index.values,
        'unique_count_flag': (wh_unique_counts > 1).astype(int), 
        'unique_value': wh_uniques.values
    })
    # print('\n\n')
    # print(wh_uc.head())
    # print('\n\n')
    t1 = time.time()
    tc += t1 - t0

    # Write intermediate results
    t0 = time.time()
    res = write_intermediate(wh_uc, key)
    t1 = time.time()
    tw += t1 - t0

    res['breakdown'] = [tr, tc, tw, (tr+tc+tw)]
    return res


def q95_stage1(key):
    [tr, tc, tw] = [0] * 3

    # Read stage 0's results
    t0 = time.time()
    wh_uc = read_data(key)
    t1 = time.time()
    tr += t1 - t0

    # Reduced groupby, including unique_count_flag == 1 or unique_value > 1
    t0 = time.time()
    
    condition1 = wh_uc.groupby('ws_order_number')['unique_count_flag'].transform('max') == 1
    
    condition2 = (wh_uc.groupby('ws_order_number')['unique_count_flag'].transform('max') == 0) & (wh_uc.groupby('ws_order_number')['unique_value'].transform('nunique') > 1)
    target_ = wh_uc[condition1 | condition2]
                   
    target_order_number = pd.DataFrame(target_['ws_order_number'].drop_duplicates())
    t1 = time.time()
    tc += t1 - t0

    # Write intermediate results
    t0 = time.time()
    res = write_intermediate(target_order_number, key)
    t1 = time.time()
    tw += t1 - t0

    res['breakdown'] = [tr, tc, tw, (tr+tc+tw)]
    return res


def q95_stage2(key):
    [tr, tc, tw] = [0] * 3

    # Read web_sales
    t0 = time.time()
    # should be read partitions
    ws = read_data(key)
    t1 = time.time()
    tr += t1 - t0

    # Select columns from web_sales
    t0 = time.time()
    wanted_columns = ['ws_order_number',
                      'ws_ext_ship_cost',
                      'ws_net_profit',
                      'ws_ship_date_sk',
                      'ws_ship_addr_sk',
                      'ws_web_site_sk',
                      'ws_warehouse_sk']
    ws_s = ws[wanted_columns]
    t1 = time.time()
    tc += t1 - t0

    # Write intermediate results
    t0 = time.time()
    res = write_intermediate(ws_s, key)
    t1 = time.time()
    tw += t1 - t0

    res['breakdown'] = [tr, tc, tw, (tr+tc+tw)]
    return res


def q95_stage3(key):
    [tr, tc, tw] = [0] * 3

    # Read web_returns
    t0 = time.time()
    wr = read_data(key)
    t1 = time.time()
    tr += t1 - t0

    # Select column wr_order_number from web_returns
    t0 = time.time()
    wr_order_number = wr[['wr_order_number']]
    t1 = time.time()
    tc += t1 - t0

    # Write intermediate results
    t0 = time.time()
    res = write_intermediate(wr_order_number, key)
    t1 = time.time()
    tw += t1 - t0

    res['breakdown'] = [tr, tc, tw, (tr+tc+tw)]
    return res


def q95_stage4(key):
    [tr, tc, tw] = [0] * 3

    # Read the results from stage 2, 3, 1 and date_dim
    t0 = time.time()
    ws = read_data(key, 0)  # stage 2
    wr = read_data(key, 1)  # stage 3
    d = read_data(key, 2)  # date_dim
    ws_wh = read_data(key, 3)  # stage 1
    t1 = time.time()
    tr += t1 - t0

    # Filter ws by ws_order_number in ws_wh and wr
    t0 = time.time()
    ws_f1 = ws.loc[ws['ws_order_number'].isin(ws_wh['ws_order_number'])]
    ws_f2 = ws_f1.loc[ws_f1['ws_order_number'].isin(wr['wr_order_number'])]    

    # Filter d_date
    dd = d[['d_date', 'd_date_sk']]
    dd_select = dd[(pd.to_datetime(dd['d_date']) > pd.to_datetime('1999-04-01')) & 
                   (pd.to_datetime(dd['d_date']) < pd.to_datetime('1999-06-01'))]
    dd_filtered = dd_select[['d_date_sk']]
    
    # Join by ws_ship_date_sk and d_date_sk
    ws_fd = ws_f2.merge(dd_filtered, left_on='ws_ship_date_sk', right_on='d_date_sk')
    del dd
    del ws_f2
    del dd_select
    del dd_filtered
    ws_fd.drop('d_date_sk', axis=1, inplace=True)
    t1 = time.time()
    tc += t1 - t0

    # Write intermediate results
    t0 = time.time()
    res = write_intermediate(ws_fd, key)
    t1 = time.time()
    tw += t1 - t0

    res['breakdown'] = [tr, tc, tw, (tr+tc+tw)]
    return res


def q95_stage5(key):
    [tr, tc, tw] = [0] * 3

    # Read customer_address
    t0 = time.time()
    ca = read_data(key)
    t1 = time.time()
    tr += t1 - t0

    # Filter ca and select ca_address_sk
    t0 = time.time()
    ca_s = ca[ca.ca_state == 'IA'][['ca_address_sk']]
    t1 = time.time()
    tc += t1 - t0

    # Write intermediate results
    t0 = time.time()
    res = write_intermediate(ca_s, key)
    t1 = time.time()
    tw += t1 - t0

    res['breakdown'] = [tr, tc, tw, (tr+tc+tw)]
    return res


def q95_stage6(key):
    [tr, tc, tw] = [0] * 3

    # Read stage 4 and 5's results and web_site
    t0 = time.time()
    ws = read_data(key, 0)
    ca = read_data(key, 1)
    web = read_data(key, 2)
    t1 = time.time()
    tr += t1 - t0

    t0 = time.time()
    # Join by ws_ship_addr_sk and ca_address_sk
    ws_j1 = ws.merge(ca, left_on='ws_ship_addr_sk', right_on='ca_address_sk')
    ws_j1.drop('ws_ship_addr_sk', axis=1, inplace=True)
    # Filter web_site by web_company_name
    web_f = web[web['web_company_name'] == 'pri'][['web_site_sk']]
    ws_j2 = ws_j1.merge(web_f, left_on='ws_web_site_sk', right_on='web_site_sk')
    target_data = ws_j2[['ws_order_number', 'ws_ext_ship_cost', 'ws_net_profit']]
    t1 = time.time()
    tc += t1 - t0

    # Write intermediate results
    t0 = time.time()
    res = write_intermediate(target_data, key)
    t1 = time.time()
    tw += t1 - t0

    res['breakdown'] = [tr, tc, tw, (tr+tc+tw)]
    return res


def q95_stage7(key):
    [tr, tc, tw] = [0] * 3

    # Read stage 6's results
    t0 = time.time()
    ws_tgt = read_data(key)
    t1 = time.time()
    tr += t1 - t0

    # Calculate the final results
    t0 = time.time()
    a1 = pd.unique(ws_tgt['ws_order_number']).size
    a2 = ws_tgt['ws_ext_ship_cost'].sum()
    a3 = ws_tgt['ws_net_profit'].sum()
    # Create the final results dataframe
    final_result = pd.DataFrame({'unique_order_number_count': [a1], 
                                 'ship_cost_sum': [a2], 
                                 'net_profit_sum': [a3]})
    t1 = time.time()
    tc += t1 - t0

    # Write final results
    t0 = time.time()
    res = write_intermediate(final_result, key)
    t1 = time.time()
    tw += t1 - t0

    res['breakdown'] = [tr, tc, tw, (tr+tc+tw)]
    return res


def invoke_q95_func(event):
    schema = merge_dicts(table_schema, q95_intermediate_schema)
    schemas = [schema] * len(event['input_address'])

    key = create_key(task_id=event['task_id'],
            input_address=event['input_address'], 
            table_name=event['table_name'], 
            schema=schemas,
            read_pattern=event['read_pattern'],
            output_address=event['output_address'],
            storage_mode=event['storage_mode'],
            num_tasks=event['num_tasks'],
            num_partitions=event['num_partitions'],
            func_id=event['func_id'])
    
    func_id = key['func_id']
    if func_id == 0:
        return q95_stage0(key)
    elif func_id == 1:
        return q95_stage1(key)
    elif func_id == 2:
        return q95_stage2(key)
    elif func_id == 3:
        return q95_stage3(key)
    elif func_id == 4:
        return q95_stage4(key)
    elif func_id == 5:
        return q95_stage5(key)
    elif func_id == 6:
        return q95_stage6(key)
    elif func_id == 7:
        return q95_stage7(key)
    else:
        raise ValueError('Invalid func_id')


if __name__ == "__main__":
    schema = merge_dicts(table_schema, q95_intermediate_schema)
    # local test

    dict_stage0 = {
        'task_id': 2,
        'input_address': ['tpcds/test-1g/web_sales'],
        'table_name': ['web_sales'],
        'read_pattern': ['read_partial_table'],
        'output_address': 'tpcds/test-1g/q95_intermediate/q95_stage0',
        'storage_mode': 's3',
        'num_tasks': 30,
        'num_partitions': [1],
        'func_id': 0
    }

    dict_stage0_local = {
        'task_id': 0,
        'input_address': ['../data/web_sales'],
        'table_name': ['web_sales'],
        'read_pattern': ['read_partial_table'],
        'output_address': '../data/q95_intermediate/q95_stage0',
        'storage_mode': 'local',
        'num_tasks': 1,
        'num_partitions': [1],
        'func_id': 0
    }


    dict_stage1 = {
        'task_id': 0,
        'input_address': 'tpcds/test-1g/q95_intermediate/q95_stage0',
        'table_name': 'stage0',
        'read_pattern': 'read_all_partitions',
        'output_address': 'tpcds/test-1g/q95_intermediate/q95_stage1',
        'storage_mode': 's3',
        'num_tasks': 1,
        'num_partitions': 30,
        'func_id': 1
    }


    dict_stage1_local = {
        'task_id': 0,
        'input_address': ['../data/q95_intermediate/q95_stage0'],
        'table_name': ['stage0'],
        'read_pattern': ['read_all_partitions'],
        'output_address': '../data/q95_intermediate/q95_stage1',
        'storage_mode': 'local',
        'num_tasks': 10,
        'num_partitions': [1],
        'func_id': 1
    }


    dict_stage2 = {
        'task_id': 0,
        'input_address': 'tpcds/test-1g/web_sales',
        'table_name': 'web_sales',
        'read_pattern': 'read_partial_table',
        'output_address': 'tpcds/test-1g/q95_intermediate/q95_stage2',
        'storage_mode': 's3',
        'num_tasks': 30,
        'func_id': 2
    }

    
    dict_stage2_local = {
        'task_id': 0,
        'input_address': 'tpcds/test-1g/web_sales',
        'table_name': 'web_sales',
        'read_pattern': 'read_partial_table',
        'output_address': 'tpcds/test-1g/q95_intermediate/q95_stage2',
        'storage_mode': 's3',
        'num_tasks': 30,
        'func_id': 2
    }


    dict_stage3 = {
        'task_id': 1,
        'input_address': 'tpcds/test-1g/web_returns',
        'table_name': 'web_returns',
        'read_pattern': 'read_partial_table',
        'output_address': 'tpcds/test-1g/q95_intermediate/q95_stage3',
        'storage_mode': 's3',
        'num_tasks': 2,
        'func_id': 3
    }

    
    dict_stage3_local = {
        'task_id': 0,
        'input_address': '../data/web_returns',
        'table_name': 'web_returns',
        'read_pattern': 'read_partial_table',
        'output_address': '../data/q95_intermediate/q95_stage3',
        'storage_mode': 'local',
        'num_tasks': 2,
        'func_id': 3
    }


    dict_stage4 = {
        'task_id': 0,
        'input_address': ['tpcds/test-1g/q95_intermediate/q95_stage2', 
                        'tpcds/test-1g/q95_intermediate/q95_stage3', 
                        'tpcds/test-1g/date_dim',
                        'tpcds/test-1g/q95_intermediate/q95_stage1'],
        'table_name': ['stage2', 'stage3', 'date_dim', 'stage1'],
        'read_pattern': ['read_multiple_partitions', 'read_all_partitions', 
                        'read_table', 'read_all_partitions'],
        'output_address': 'tpcds/test-1g/q95_intermediate/q95_stage4',
        'storage_mode': 's3',
        'num_tasks': 30,
        'num_partitions': [30, 2, 1, 1],
        'func_id': 4
    }

    dict_stage4_local = {
        'task_id': 0,
        'input_address': ['tpcds/test-1g/q95_intermediate/q95_stage2', 
                        'tpcds/test-1g/q95_intermediate/q95_stage3', 
                        'tpcds/test-1g/date_dim',
                        'tpcds/test-1g/q95_intermediate/q95_stage1'],
        'table_name': ['stage2', 'stage3', 'date_dim', 'stage1'],
        'read_pattern': ['read_multiple_partitions', 'read_all_partitions', 
                        'read_table', 'read_all_partitions'],
        'output_address': 'tpcds/test-1g/q95_intermediate/q95_stage4',
        'storage_mode': 's3',
        'num_tasks': 30,
        'num_partitions': [30, 2, 1, 1],
        'func_id': 4
    }
    

    dict_stage5 = {
        'task_id': 0,
        'input_address': 'tpcds/test-1g/customer_address',
        'table_name': 'customer_address',
        'read_pattern': 'read_table',
        'output_address': 'tpcds/test-1g/q95_intermediate/q95_stage5',
        'storage_mode': 's3',
        'num_tasks': 1,
        'func_id': 5
    }

    
    dict_stage5_local = {
        'task_id': 0,
        'input_address': '../data/customer_address',
        'table_name': 'customer_address',
        'read_pattern': 'read_partial_table',
        'output_address': '../data/q95_intermediate/q95_stage5',
        'storage_mode': 'local',
        'num_tasks': 1,
        'func_id': 5
    }


    dict_stage6 = {
        'task_id': 0,
        'input_address': ['tpcds/test-1g/q95_intermediate/q95_stage4', 
                        'tpcds/test-1g/q95_intermediate/q95_stage5', 
                        'tpcds/test-1g/web_site'],
        'table_name': ['stage4', 'stage5', 'web_site'],
        'read_pattern': ['read_multiple_partitions', 'read_all_partitions', 'read_table'],
        'output_address': 'tpcds/test-1g/q95_intermediate/q95_stage6',
        'storage_mode': 's3',
        'num_tasks': 1,
        'num_partitions': [1, 1, 1],
        'func_id': 6
    }

    
    dict_stage6_local = {
        'task_id': 0,
        'input_address': ['../data/q95_intermediate/q95_stage4', 
                        '../data/q95_intermediate/q95_stage5', 
                        '../data/web_site'],
        'table_name': ['stage4', 'stage5', 'web_site'],
        'read_pattern': ['read_multiple_partitions', 'read_all_partitions', 'read_table'],
        'output_address': '../data/q95_intermediate/q95_stage6',
        'storage_mode': 'local',
        'num_tasks': 1,
        'num_partitions': [1, 1, 1],
        'func_id': 6
    }


    dict_stage7 = {
        'task_id': 0,
        'input_address': 'tpcds/test-1g/q95_intermediate/q95_stage6',
        'table_name': 'stage6',
        'read_pattern': 'read_all_partitions',
        'output_address': 'tpcds/test-1g/q95_intermediate/q95_stage7',
        'storage_mode': 's3',
        'num_tasks': 1,
        'num_partitions': 1,
        'func_id': 7
    }


    dict_stage7_local = {
        'task_id': 0,
        'input_address': '../data/q95_intermediate/q95_stage6',
        'table_name': 'stage6',
        'read_pattern': 'read_all_partitions',
        'output_address': '../data/q95_intermediate/q95_stage7',
        'storage_mode': 'local',
        'num_tasks': 1,
        'num_partitions': 1,
        'func_id': 7
    }


    res = invoke_q95_func(dict_stage1_local)
    print('\n\n')
    print(res)