import pickle
import pandas as pd
import os
import numpy as np
import datetime
import sys
import subprocess

import time

from io import StringIO
import boto3
from io import BytesIO
from multiprocessing.pool import ThreadPool

import logging
import random

s3_bucket_default = 'serverless-bound'

table_schema = {
    'dbgen_version': {
        'dv_version': 'string',
        'dv_create_date': 'date',
        'dv_create_time': 'string',
        'dv_cmdline_args': 'string'
    },
    'customer_address': {
        'ca_address_sk': 'int64',
        'ca_address_id': 'string',
        'ca_street_number': 'string',
        'ca_street_name': 'string',
        'ca_street_type': 'string',
        'ca_suite_number': 'string',
        'ca_city': 'string',
        'ca_county': 'string',
        'ca_state': 'string',
        'ca_zip': 'string',
        'ca_country': 'string',
        'ca_gmt_offset': 'float32',
        'ca_location_type': 'string'
    },
    'customer_demographics': {
        'cd_demo_sk': 'int64',
        'cd_gender': 'string',
        'cd_marital_status': 'string',
        'cd_education_status': 'string',
        'cd_purchase_estimate': 'int64',
        'cd_credit_rating': 'string',
        'cd_dep_count': 'int64',
        'cd_dep_employed_count': 'int64',
        'cd_dep_college_count': 'int64'
    },
    'date_dim': {
        'd_date_sk': 'int64',
        'd_date_id': 'string',
        'd_date': 'date',
        'd_month_seq': 'int64',
        'd_week_seq': 'int64',
        'd_quarter_seq': 'int64',
        'd_year': 'int64',
        'd_dow': 'int64',
        'd_moy': 'int64',
        'd_dom': 'int64',
        'd_qoy': 'int64',
        'd_fy_year': 'int64',
        'd_fy_quarter_seq': 'int64',
        'd_fy_week_seq': 'int64',
        'd_day_name': 'string',
        'd_quarter_name': 'string',
        'd_holiday': 'string',
        'd_weekend': 'string',
        'd_following_holiday': 'string',
        'd_first_dom': 'int64',
        'd_last_dom': 'int64',
        'd_same_day_ly': 'int64',
        'd_same_day_lq': 'int64',
        'd_current_day': 'string',
        'd_current_week': 'string',
        'd_current_month': 'string',
        'd_current_quarter': 'string',
        'd_current_year': 'string'
    },
    'warehouse': {
        'w_warehouse_sk': 'int64',
        'w_warehouse_id': 'string',
        'w_warehouse_name': 'string',
        'w_warehouse_sq_ft': 'int64',
        'w_street_number': 'string',
        'w_street_name': 'string',
        'w_street_type': 'string',
        'w_suite_number': 'string',
        'w_city': 'string',
        'w_county': 'string',
        'w_state': 'string',
        'w_zip': 'string',
        'w_country': 'string',
        'w_gmt_offset': 'float32'
    },
    'ship_mode': {
        'sm_ship_mode_sk': 'int64',
        'sm_ship_mode_id': 'string',
        'sm_type': 'string',
        'sm_code': 'string',
        'sm_carrier': 'string',
        'sm_contract': 'string'
    },
    'time_dim': {
        't_time_sk': 'int64',
        't_time_id': 'string',
        't_time': 'int64',
        't_hour': 'int64',
        't_minute': 'int64',
        't_second': 'int64',
        't_am_pm': 'string',
        't_shift': 'string',
        't_sub_shift': 'string',
        't_meal_time': 'string'
    },
    'reason': {
        'r_reason_sk': 'int64',
        'r_reason_id': 'string',
        'r_reason_desc': 'string'
    },
    'income_band': {
        'ib_income_band_sk': 'int64',
        'ib_lower_bound': 'int64',
        'ib_upper_bound': 'int64'
    },
    'item': {
        'i_item_sk': 'int64',
        'i_item_id': 'string',
        'i_rec_start_date': 'date',
        'i_rec_end_date': 'date',
        'i_item_desc': 'string',
        'i_current_price': 'float32',
        'i_wholesale_cost': 'float32',
        'i_brand_id': 'int64',
        'i_brand': 'string',
        'i_class_id': 'int64',
        'i_class': 'string',
        'i_category_id': 'int64',
        'i_category': 'string',
        'i_manufact_id': 'int64',
        'i_manufact': 'string',
        'i_size': 'string',
        'i_formulation': 'string',
        'i_color': 'string',
        'i_units': 'string',
        'i_container': 'string',
        'i_manager_id': 'int64',
        'i_product_name': 'string'
    },
    'store': {
        's_store_sk': 'int64',
        's_store_id': 'string',
        's_rec_start_date': 'date',
        's_rec_end_date': 'date',
        's_closed_date_sk': 'int64',
        's_store_name': 'string',
        's_number_employees': 'int64',
        's_floor_space': 'int64',
        's_hours': 'string',
        's_manager': 'string',
        's_market_id': 'int64',
        's_geography_class': 'string',
        's_market_desc': 'string',
        's_market_manager': 'string',
        's_division_id': 'int64',
        's_division_name': 'string',
        's_company_id': 'int64',
        's_company_name': 'string',
        's_street_number': 'string',
        's_street_name': 'string',
        's_street_type': 'string',
        's_suite_number': 'string',
        's_city': 'string',
        's_county': 'string',
        's_state': 'string',
        's_zip': 'string',
        's_country': 'string',
        's_gmt_offset': 'float32',
        's_tax_precentage': 'float32'
    },
    'call_center': {
        'cc_call_center_sk': 'int64',
        'cc_call_center_id': 'string',
        'cc_rec_start_date': 'date',
        'cc_rec_end_date': 'date',
        'cc_closed_date_sk': 'int64',
        'cc_open_date_sk': 'int64',
        'cc_name': 'string',
        'cc_class': 'string',
        'cc_employees': 'int64',
        'cc_sq_ft': 'int64',
        'cc_hours': 'string',
        'cc_manager': 'string',
        'cc_mkt_id': 'int64',
        'cc_mkt_class': 'string',
        'cc_mkt_desc': 'string',
        'cc_market_manager': 'string',
        'cc_division': 'int64',
        'cc_division_name': 'string',
        'cc_company': 'int64',
        'cc_company_name': 'string',
        'cc_street_number': 'string',
        'cc_street_name': 'string',
        'cc_street_type': 'string',
        'cc_suite_number': 'string',
        'cc_city': 'string',
        'cc_county': 'string',
        'cc_state': 'string',
        'cc_zip': 'string',
        'cc_country': 'string',
        'cc_gmt_offset': 'float32',
        'cc_tax_percentage': 'float32'
    },
    'customer': {
        'c_customer_sk': 'int64',
        'c_customer_id': 'string',
        'c_current_cdemo_sk': 'int64',
        'c_current_hdemo_sk': 'int64',
        'c_current_addr_sk': 'int64',
        'c_first_shipto_date_sk': 'int64',
        'c_first_sales_date_sk': 'int64',
        'c_salutation': 'string',
        'c_first_name': 'string',
        'c_last_name': 'string',
        'c_preferred_cust_flag': 'string',
        'c_birth_day': 'int64',
        'c_birth_month': 'int64',
        'c_birth_year': 'int64',
        'c_birth_country': 'string',
        'c_login': 'string',
        'c_email_address': 'string',
        'c_last_review_date': 'string'
    },
    'web_site': {
        'web_site_sk': 'int64',
        'web_site_id': 'string',
        'web_rec_start_date': 'date',
        'web_rec_end_date': 'date',
        'web_name': 'string',
        'web_open_date_sk': 'int64',
        'web_close_date_sk': 'int64',
        'web_class': 'string',
        'web_manager': 'string',
        'web_mkt_id': 'int64',
        'web_mkt_class': 'string',
        'web_mkt_desc': 'string',
        'web_market_manager': 'string',
        'web_company_id': 'int64',
        'web_company_name': 'string',
        'web_street_number': 'string',
        'web_street_name': 'string',
        'web_street_type': 'string',
        'web_suite_number': 'string',
        'web_city': 'string',
        'web_county': 'string',
        'web_state': 'string',
        'web_zip': 'string',
        'web_country': 'string',
        'web_gmt_offset': 'float32',
        'web_tax_percentage': 'float32'
    },
    'store_returns': {
        'sr_returned_date_sk': 'int64',
        'sr_return_time_sk': 'int64',
        'sr_item_sk': 'int64',
        'sr_customer_sk': 'int64',
        'sr_cdemo_sk': 'int64',
        'sr_hdemo_sk': 'int64',
        'sr_addr_sk': 'int64',
        'sr_store_sk': 'int64',
        'sr_reason_sk': 'int64',
        'sr_ticket_number': 'int64',
        'sr_return_quantity': 'int64',
        'sr_return_amt': 'float32',
        'sr_return_tax': 'float32',
        'sr_return_amt_inc_tax': 'float32',
        'sr_fee': 'float32',
        'sr_return_ship_cost': 'float32',
        'sr_refunded_cash': 'float32',
        'sr_reversed_charge': 'float32',
        'sr_store_credit': 'float32',
        'sr_net_loss': 'float32'
    },
    'household_demographics': {
        'hd_demo_sk': 'int64',
        'hd_income_band_sk': 'int64',
        'hd_buy_potential': 'string',
        'hd_dep_count': 'int64',
        'hd_vehicle_count': 'int64'
    },
    'web_page': {
        'wp_web_page_sk': 'int64',
        'wp_web_page_id': 'string',
        'wp_rec_start_date': 'date',
        'wp_rec_end_date': 'date',
        'wp_creation_date_sk': 'int64',
        'wp_access_date_sk': 'int64',
        'wp_autogen_flag': 'string',
        'wp_customer_sk': 'int64',
        'wp_url': 'string',
        'wp_type': 'string',
        'wp_char_count': 'int64',
        'wp_link_count': 'int64',
        'wp_image_count': 'int64',
        'wp_max_ad_count': 'int64'
    },
    'promotion': {
        'p_promo_sk': 'int64',
        'p_promo_id': 'string',
        'p_start_date_sk': 'int64',
        'p_end_date_sk': 'int64',
        'p_item_sk': 'int64',
        'p_cost': 'float64',
        'p_response_target': 'int64',
        'p_promo_name': 'string',
        'p_channel_dmail': 'string',
        'p_channel_email': 'string',
        'p_channel_catalog': 'string',
        'p_channel_tv': 'string',
        'p_channel_radio': 'string',
        'p_channel_press': 'string',
        'p_channel_event': 'string',
        'p_channel_demo': 'string',
        'p_channel_details': 'string',
        'p_purpose': 'string',
        'p_discount_active': 'string'
    },
    'catalog_page': {
        'cp_catalog_page_sk': 'int64',
        'cp_catalog_page_id': 'string',
        'cp_start_date_sk': 'int64',
        'cp_end_date_sk': 'int64',
        'cp_department': 'string',
        'cp_catalog_number': 'int64',
        'cp_catalog_page_number': 'int64',
        'cp_description': 'string',
        'cp_type': 'string'
    },
    'inventory': {
        'inv_date_sk': 'int64',
        'inv_item_sk': 'int64',
        'inv_warehouse_sk': 'int64',
        'inv_quantity_on_hand': 'int64'
    },
    'catalog_returns': {
        'cr_returned_date_sk': 'int64',
        'cr_returned_time_sk': 'int64',
        'cr_item_sk': 'int64',
        'cr_refunded_customer_sk': 'int64',
        'cr_refunded_cdemo_sk': 'int64',
        'cr_refunded_hdemo_sk': 'int64',
        'cr_refunded_addr_sk': 'int64',
        'cr_returning_customer_sk': 'int64',
        'cr_returning_cdemo_sk': 'int64',
        'cr_returning_hdemo_sk': 'int64',
        'cr_returning_addr_sk': 'int64',
        'cr_call_center_sk': 'int64',
        'cr_catalog_page_sk': 'int64',
        'cr_ship_mode_sk': 'int64',
        'cr_warehouse_sk': 'int64',
        'cr_reason_sk': 'int64',
        'cr_order_number': 'int64',
        'cr_return_quantity': 'int64',
        'cr_return_amount': 'float32',
        'cr_return_tax': 'float32',
        'cr_return_amt_inc_tax': 'float32',
        'cr_fee': 'float32',
        'cr_return_ship_cost': 'float32',
        'cr_refunded_cash': 'float32',
        'cr_reversed_charge': 'float32',
        'cr_store_credit': 'float32',
        'cr_net_loss': 'float32'
    },
    'web_returns': {
        'wr_returned_date_sk': 'int64',
        'wr_returned_time_sk': 'int64',
        'wr_item_sk': 'int64',
        'wr_refunded_customer_sk': 'int64',
        'wr_refunded_cdemo_sk': 'int64',
        'wr_refunded_hdemo_sk': 'int64',
        'wr_refunded_addr_sk': 'int64',
        'wr_returning_customer_sk': 'int64',
        'wr_returning_cdemo_sk': 'int64',
        'wr_returning_hdemo_sk': 'int64',
        'wr_returning_addr_sk': 'int64',
        'wr_web_page_sk': 'int64',
        'wr_reason_sk': 'int64',
        'wr_order_number': 'int64',
        'wr_return_quantity': 'int64',
        'wr_return_amt': 'float32',
        'wr_return_tax': 'float32',
        'wr_return_amt_inc_tax': 'float32',
        'wr_fee': 'float32',
        'wr_return_ship_cost': 'float32',
        'wr_refunded_cash': 'float32',
        'wr_reversed_charge': 'float32',
        'wr_account_credit': 'float32',
        'wr_net_loss': 'float32'
    },
    'web_sales': {
        'ws_sold_date_sk': 'int64',
        'ws_sold_time_sk': 'int64',
        'ws_ship_date_sk': 'int64',
        'ws_item_sk': 'int64',
        'ws_bill_customer_sk': 'int64',
        'ws_bill_cdemo_sk': 'int64',
        'ws_bill_hdemo_sk': 'int64',
        'ws_bill_addr_sk': 'int64',
        'ws_ship_customer_sk': 'int64',
        'ws_ship_cdemo_sk': 'int64',
        'ws_ship_hdemo_sk': 'int64',
        'ws_ship_addr_sk': 'int64',
        'ws_web_page_sk': 'int64',
        'ws_web_site_sk': 'int64',
        'ws_ship_mode_sk': 'int64',
        'ws_warehouse_sk': 'int64',
        'ws_promo_sk': 'int64',
        'ws_order_number': 'int64',
        'ws_quantity': 'int64',
        'ws_wholesale_cost': 'float32',
        'ws_list_price': 'float32',
        'ws_sales_price': 'float32',
        'ws_ext_discount_amt': 'float32',
        'ws_ext_sales_price': 'float32',
        'ws_ext_wholesale_cost': 'float32',
        'ws_ext_list_price': 'float32',
        'ws_ext_tax': 'float32',
        'ws_coupon_amt': 'float32',
        'ws_ext_ship_cost': 'float32',
        'ws_net_paid': 'float32',
        'ws_net_paid_inc_tax': 'float32',
        'ws_net_paid_inc_ship': 'float32',
        'ws_net_paid_inc_ship_tax': 'float32',
        'ws_net_profit': 'float32'
    },
    'catalog_sales': {
        'cs_sold_date_sk': 'int64',
        'cs_sold_time_sk': 'int64',
        'cs_ship_date_sk': 'int64',
        'cs_bill_customer_sk': 'int64',
        'cs_bill_cdemo_sk': 'int64',
        'cs_bill_hdemo_sk': 'int64',
        'cs_bill_addr_sk': 'int64',
        'cs_ship_customer_sk': 'int64',
        'cs_ship_cdemo_sk': 'int64',
        'cs_ship_hdemo_sk': 'int64',
        'cs_ship_addr_sk': 'int64',
        'cs_call_center_sk': 'int64',
        'cs_catalog_page_sk': 'int64',
        'cs_ship_mode_sk': 'int64',
        'cs_warehouse_sk': 'int64',
        'cs_item_sk': 'int64',
        'cs_promo_sk': 'int64',
        'cs_order_number': 'int64',
        'cs_quantity': 'int64',
        'cs_wholesale_cost': 'float32',
        'cs_list_price': 'float32',
        'cs_sales_price': 'float32',
        'cs_ext_discount_amt': 'float32',
        'cs_ext_sales_price': 'float32',
        'cs_ext_wholesale_cost': 'float32',
        'cs_ext_list_price': 'float32',
        'cs_ext_tax': 'float32',
        'cs_coupon_amt': 'float32',
        'cs_ext_ship_cost': 'float32',
        'cs_net_paid': 'float32',
        'cs_net_paid_inc_tax': 'float32',
        'cs_net_paid_inc_ship': 'float32',
        'cs_net_paid_inc_ship_tax': 'float32',
        'cs_net_profit': 'float32'
    },
    'store_sales': {
        'ss_sold_date_sk': 'int64',
        'ss_sold_time_sk': 'int64',
        'ss_item_sk': 'int64',
        'ss_customer_sk': 'int64',
        'ss_cdemo_sk': 'int64',
        'ss_hdemo_sk': 'int64',
        'ss_addr_sk': 'int64',
        'ss_store_sk': 'int64',
        'ss_promo_sk': 'int64',
        'ss_ticket_number': 'int64',
        'ss_quantity': 'int64',
        'ss_wholesale_cost': 'float32',
        'ss_list_price': 'float32',
        'ss_sales_price': 'float32',
        'ss_ext_discount_amt': 'float32',
        'ss_ext_sales_price': 'float32',
        'ss_ext_wholesale_cost': 'float32',
        'ss_ext_list_price': 'float32',
        'ss_ext_tax': 'float32',
        'ss_coupon_amt': 'float32',
        'ss_net_paid': 'float32',
        'ss_net_paid_inc_tax': 'float32',
        'ss_net_profit': 'float32'
    }
}


fillna_dict = {
    pd.Int32Dtype() : 0,
    pd.Int64Dtype() : 0,
    pd.Float32Dtype() : 0.,
    pd.Float64Dtype() : 0.,
    pd.StringDtype() : 'Nan'
}


def get_pd_type(typename):
    if typename == "date":
        return datetime.datetime
    if "decimal" in typename:
        return pd.Float64Dtype()
    if typename == "int32":
        return pd.Int32Dtype()
    if typename == "int64":
        return pd.Int64Dtype()
    if typename == "float32":
        return pd.Float32Dtype()
    if typename == "float64":
        return pd.Float64Dtype()
    if typename == "string":
        return pd.StringDtype()
    raise Exception("Not supported type in pandas: " + typename)

def pd_tpye_to_np_type(pd_type):
    return str(pd_type).lower()
    # if pd_type == pd.Int32Dtype():
    #     return 'Int32'
    # if pd_type == pd.Int64Dtype():
    #     return 'Int64'
    # if pd_type == datetime.datetime:
    #     return 'date'
    # if pd_type == pd.Float32Dtype():
    #     return 'Float32'
    # if pd_type == pd.Float64Dtype():
    #     return 'Float64'
    # if pd_type == pd.StringDtype():
    #     return 'String'
    # raise Exception("Not supported type in pandas")


def get_np_type(typename):
    if typename == "date":
        return np.datetime64
    if "decimal" in typename:
        return np.dtype('float')
    if typename == "int32":
        return np.dtype('int32')
    if typename == "int64":
        return np.dtype('int64')
    if typename == "float32":
        return np.dtype('float32')
    if typename == "string":
        return np.dtype(typename)
    raise Exception("Not supported type in numpy: " + typename)

# only for local file
def count_lines_wc(file_path):
    result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
    
    line_count = int(result.stdout.split()[0])
    return line_count

def merge_dicts(*dicts):
    result_dict = {}
    for dictionary in dicts:
        duplicate_keys = set(dictionary.keys()) & set(result_dict.keys())
        if duplicate_keys:
            raise ValueError(f"Duplicate keys: {duplicate_keys}")
        result_dict.update(dictionary)
    return result_dict

def genrate_meta_data(df):
    assert isinstance(df, pd.DataFrame)
    pass

def read_local_table(key):
    location = key['input_address'] + key['suffix']
    names = list(key['column_names'])
    dtypes = key['dtypes']
    parse_dates = []
    for d in dtypes:
        if dtypes[d] == datetime.datetime or dtypes[d] == np.datetime64:
            parse_dates.append(d)
            dtypes[d] = pd.StringDtype()
    part_data = pd.read_table(location, 
                              delimiter="|", 
                              header=None, 
                              names=names,
                              usecols=range(len(names)), 
                              dtype=dtypes, 
                              na_values='-')
                            #   parse_dates=parse_dates)
    print(part_data.info())
    
    fill_dict = {}
    for k, v in dtypes.items():
        fill_dict[k] = fillna_dict[v]
    part_data = part_data.fillna(fill_dict)

    return part_data


def read_s3_table(key, s3_client=None):
    loc = key['input_address']
    names = list(key['column_names'])
    dtypes = key['dtypes']
    parse_dates = []
    for d in dtypes:
        if dtypes[d] == datetime.datetime or dtypes[d] == np.datetime64:
            parse_dates.append(d)
            dtypes[d] = pd.StringDtype()
    if s3_client == None:
        s3_client = boto3.client("s3")
    data = []
    if isinstance(key['input_address'], str):
        loc = key['input_address'] + key['suffix']
        obj = s3_client.get_object(Bucket=s3_bucket_default, Key=loc)['Body'].read()
        data.append(obj)
    else:
        for loc in key['input_address']:
            loc_ = loc + key['suffix']
            obj = s3_client.get_object(Bucket=s3_bucket_default, Key=loc_)['Body'].read()
            data.append(obj)
    part_data = pd.read_table(BytesIO(b"".join(data)),
                              delimiter="|", 
                              header=None, 
                              names=names,
                              usecols=range(len(names)), 
                              dtype=dtypes, 
                              na_values = "-")
                            #   parse_dates=parse_dates)
    print(part_data.info())

    fill_dict = {}
    for k, v in dtypes.items():
        fill_dict[k] = fillna_dict[v]
    part_data = part_data.fillna(fill_dict)

    return part_data

# read the index-th 1/partitions part of the table
def read_local_partial_table(key, index, partitions):
    assert isinstance(key, dict)
    assert isinstance(index, int)
    assert isinstance(partitions, int)
    
    location = key['input_address'] + key['suffix']
    names = list(key['column_names'])
    dtypes = key['dtypes']

    nrows = count_lines_wc(location)
    part_size = nrows // partitions
    start_row = index * part_size
    if index == partitions - 1:
        end_row = nrows - 1
    else:
        end_row = (index + 1) * part_size - 1
    
    parse_dates = []
    for d in dtypes:
        if dtypes[d] == datetime.datetime or dtypes[d] == np.datetime64:
            parse_dates.append(d)
            dtypes[d] = pd.StringDtype()
    part_data = pd.read_table(location, 
                              delimiter="|", 
                              header=None, 
                              names=names,
                              usecols=range(len(names)), 
                              dtype=dtypes, 
                              na_values='-',
                            #   parse_dates=parse_dates,
                              skiprows=lambda x: x < start_row or x > end_row)
    print(part_data.info())
    
    fill_dict = {}
    for k, v in dtypes.items():
        fill_dict[k] = fillna_dict[v]
    part_data = part_data.fillna(fill_dict)

    return part_data
    
    
# read the index-th 1/partitions part of the table
def read_s3_partial_table(key, index, partitions, s3_client=None):
    assert isinstance(key, dict)
    assert isinstance(index, int)
    assert isinstance(partitions, int)
    
    # read meta data to get the bytes range
    loc = key['input_address'] + key['suffix']
    suffix = '_meta'
    meta_loc = loc[:loc.rfind('.')] + suffix + loc[loc.rfind('.'):]
    meta_dtypes = {'num_bytes': pd.Int64Dtype()}
    if s3_client == None:
        s3_client = boto3.client("s3")
    
    obj = s3_client.get_object(Bucket=s3_bucket_default, Key=meta_loc)['Body'].read()
    meta_data = pd.read_table(BytesIO(obj),
                              delimiter="|", 
                              header=None, 
                              names=['num_bytes'],
                              usecols=range(1), 
                              dtype=meta_dtypes, 
                              na_values = "-")
    
    start_bytes = 0
    end_bytes = 0
    row_count = meta_data.shape[0]
    part_size = row_count // partitions
    start_row = index * part_size
    if index == partitions - 1:
        end_row = row_count - 1
    else:
        end_row = (index + 1) * part_size - 1

    if start_row == 0:
        start_bytes = 0
    else:
        start_bytes = meta_data.iloc[start_row - 1]['num_bytes']
        
    end_bytes = meta_data.iloc[end_row]['num_bytes']
    
    # bytes_range is inclusive on both ends
    bytes_range = "bytes=" + str(start_bytes) + "-" + str(end_bytes - 1)
    
    # read partial s3 files
    names = list(key['column_names'])
    dtypes = key['dtypes']
    parse_dates = []
    for d in dtypes:
        if dtypes[d] == datetime.datetime or dtypes[d] == np.datetime64:
            parse_dates.append(d)
            dtypes[d] = pd.StringDtype()
    if s3_client == None:
        s3_client = boto3.client("s3")
    data = []
    if isinstance(key['input_address'], str):
        loc = key['input_address'] + key['suffix']
        obj = s3_client.get_object(Bucket=s3_bucket_default, Key=loc, Range=bytes_range)['Body'].read()
        data.append(obj)
    else:
        for loc in key['input_address']:
            loc_ = loc + key['suffix']
            obj = s3_client.get_object(Bucket=s3_bucket_default, Key=loc_, Range=bytes_range)['Body'].read()
            data.append(obj)
    part_data = pd.read_table(BytesIO(b"".join(data)),
                              delimiter="|", 
                              header=None, 
                              names=names,
                              usecols=range(len(names)), 
                              dtype=dtypes, 
                              na_values = "-")
                            #   parse_dates=parse_dates)
    print(part_data.info())

    fill_dict = {}
    for k, v in dtypes.items():
        fill_dict[k] = fillna_dict[v]
    part_data = part_data.fillna(fill_dict)

    return part_data


def write_local_intermediate(table, output_loc):
    assert isinstance(table, pd.DataFrame)
    assert isinstance(output_loc, str)

    output_info = {}
    if 'bin' in table.columns:
        slt_columns = table.columns.delete(table.columns.get_loc('bin'))
    else:
        slt_columns = table.columns

    table.to_csv(output_loc, sep="|", header=False, index=False, columns=slt_columns)
    output_info['output_address'] = output_loc
    output_info['column_names'] = slt_columns.tolist()
    type_list = table.dtypes[slt_columns].tolist()
    output_info['dtypes'] = [pd_tpye_to_np_type(i) for i in type_list]

    return output_info


def write_s3_intermediate(table, output_loc, s3_client=None):
    assert isinstance(table, pd.DataFrame)
    assert isinstance(output_loc, str)

    csv_buffer = BytesIO()
    if 'bin' in table.columns:
        slt_columns = table.columns.delete(table.columns.get_loc('bin'))
    else:
        slt_columns = table.columns
    table.to_csv(csv_buffer, sep="|", header=False, index=False, columns=slt_columns)
    if s3_client == None:
        s3_client = boto3.client('s3')
        
    s3_client.put_object(Bucket=s3_bucket_default,
                         Key=output_loc,
                         Body=csv_buffer.getvalue())
    output_info = {}
    output_info['output_address'] = output_loc
    output_info['column_names'] = slt_columns.tolist()
    type_list = table.dtypes[slt_columns].tolist()
    output_info['dtypes'] = [pd_tpye_to_np_type(i) for i in type_list]

    return output_info


# Do not fillno on intermediate data
def read_local_intermediate(key):
    location = key['input_address']
    names = list(key['column_names'])
    dtypes = key['dtypes']
    parse_dates = []
    for d in dtypes:
        if dtypes[d] == datetime.datetime or dtypes[d] == np.datetime64:
            parse_dates.append(d)
            dtypes[d] = pd.StringDtype()
    part_data = pd.read_table(location, 
                              delimiter="|", 
                              header=None, 
                              names=names,
                              dtype=dtypes)
                            #   parse_dates=parse_dates)
    return part_data


def read_s3_intermediate(key, s3_client=None):
    location = key['input_address']
    names = list(key['column_names'])
    dtypes = key['dtypes']
    parse_dates = []
    for d in dtypes:
        if dtypes[d] == datetime.datetime or dtypes[d] == np.datetime64:
            parse_dates.append(d)
            dtypes[d] = pd.StringDtype()
    if s3_client == None:
        s3_client = boto3.client("s3")
    obj = s3_client.get_object(Bucket=s3_bucket_default, Key=location)
    part_data = pd.read_table(BytesIO(obj['Body'].read()), 
                              delimiter="|", 
                              header=None, 
                              names=names,
                              dtype=dtypes)
                            #   parse_dates=parse_dates)
    return part_data


# Used for reading multiple partitions, not used for all-to-all shuffle 
def get_start_end_index(task_id, num_tasks, num_partitions):
    assert num_partitions >= num_tasks
    if task_id >= num_tasks:
        # read all partitions
        assert num_tasks == 1
        return 0, num_partitions - 1
        
    num_parts_per_task = num_partitions // num_tasks
    num_remain_parts = num_partitions % num_tasks
    if task_id < num_remain_parts:
        start_index = task_id * (num_parts_per_task + 1)
        end_index = start_index + num_parts_per_task + 1
    else:
        start_index = task_id * num_parts_per_task + num_remain_parts
        end_index = start_index + num_parts_per_task
    return start_index, end_index


def read_local_multiple_partitions(key):
    start_index, end_index = get_start_end_index(key['task_id'], key['num_tasks'], key['num_partitions'])
    k = {}
    k['column_names'] = key['column_names']
    k['dtypes'] = key['dtypes']
    ds = []
    for i in range(start_index, end_index):
        k['input_address'] = key['input_address'] + '_' + str(i) + key['suffix']
        d = read_local_intermediate(k)
        ds.append(d)
    return pd.concat(ds)


def read_s3_multiple_partitions(key, threadpool=False):
    start_index, end_index = get_start_end_index(key['task_id'], key['num_tasks'], key['num_partitions'])
    k = {}
    k['column_names'] = key['column_names']
    k['dtypes'] = key['dtypes']

    ds = [i for i in range(start_index, end_index)]
    s3_client = boto3.client("s3")

    if threadpool:
        def read_work(partition_index):
            k['input_address'] = key['input_address'] + '_' + str(partition_index) + key['suffix']
            d = read_s3_intermediate(k, s3_client)
            ds[partition_index - start_index] = d
        
        read_pool = ThreadPool(1)
        read_pool.map(read_work, range(start_index, end_index))
        read_pool.close()
        read_pool.join()
    else:
        for i in range(start_index, end_index):
            k['input_address'] = key['input_address'] + '_' + str(i) + key['suffix']
            d = read_s3_intermediate(k, s3_client)
            ds[i - start_index] = d

    return pd.concat(ds)


# local is for debugging
def read_table(key, storage_mode = 's3'):
    if storage_mode == "local":
        return read_local_table(key)
    elif storage_mode == "s3":
        return read_s3_table(key)
    else:
        raise Exception("Invalid storage mode")


def read_partial_table(key, storage_mode = 's3'):
    if storage_mode == "local":
        return read_local_partial_table(key, key['task_id'], key['num_tasks'])
    elif storage_mode == "s3":
        return read_s3_partial_table(key, key['task_id'], key['num_tasks'])
    else:
        raise Exception("Invalid storage mode")


def read_multiple_partitions(key, storage_mode = 's3'):
    if storage_mode == "local":
        return read_local_multiple_partitions(key)
    elif storage_mode == "s3":
        return read_s3_multiple_partitions(key)
    else:
        raise Exception("Invalid storage mode")


# Read data by the idx-th task
def read_data(key, idx=0):
    k = None
    if isinstance(key['input_address'], str):
        k = key
    else:
        assert idx < len(key['input_address'])
        k = {}
        k['task_id'] = key['task_id']
        k['num_tasks'] = key['num_tasks']
        k['input_address'] = key['input_address'][idx]
        k['column_names'] = key['column_names'][idx]
        k['dtypes'] = key['dtypes'][idx]
        k['read_pattern'] = key['read_pattern'][idx]
        k['num_partitions'] = key['num_partitions'][idx]
        k['output_address'] = key['output_address']
        k['storage_mode'] = key['storage_mode']
        k['suffix'] = key['suffix']
    
    if k['read_pattern'] == 'read_table':
        return read_table(k, k['storage_mode'])
    elif k['read_pattern'] == 'read_partial_table':
        return read_partial_table(k, k['storage_mode'])
    elif k['read_pattern'] == 'read_multiple_partitions':
        return read_multiple_partitions(k, k['storage_mode'])
    elif k['read_pattern'] == 'read_all_partitions':
        k['num_tasks'] = 1
        return read_multiple_partitions(k, k['storage_mode'])
    else:
        raise Exception("Invalid read pattern")


def write_intermediate(table, key):
    # Note that the file output address format is as follows.
    output_loc = key['output_address'] + '_' + str(key['task_id']) + key['suffix']
    if key['storage_mode'] == "local":
        return write_local_intermediate(table, output_loc)
    elif key['storage_mode'] == "s3":
        return write_s3_intermediate(table, output_loc)
    else:
        raise Exception("Invalid storage mode")


'''
    Create a key for a serverless task
    @param: 
        task_id: int, task id
        input_address: str or list of str, input file address
        table_name: str or list of str, table (schema) name
        schema: dict or list of dict, schema of the table
        read_pattern: str or list of str, read pattern, including 'read_table', 
            'read_partial_table', 'read_multiple_partitions', 'read_all_partitions'
        output_address: str, output file address
        num_tasks: int, number of tasks in current stage
        num_partitions: int or list of int, number of partitions to read, only used when 
            read_pattern is or contains 'read_multiple_partitions' or 'read_all_partitions'. 
            The default value is 1. If num_partitions is a list, then it should have default 
            value 1 for all other read patterns.
        storage_mode: str, storage mode
    @return:
        key: dict, a key for a serverless task
'''
def create_key(task_id, input_address, table_name, read_pattern, output_address, 
               num_tasks=1, num_partitions=1, storage_mode='s3', suffix='.dat', schema=None, func_id=None):
    assert isinstance(task_id, int) and task_id >= 0
    assert isinstance(num_tasks, int) and num_tasks > 0 and task_id < num_tasks

    if isinstance(input_address, str):
        assert isinstance(table_name, str) and isinstance(schema, dict) and \
               table_name in schema and isinstance(read_pattern, str) and read_pattern in \
               ['read_table', 'read_partial_table', 'read_multiple_partitions', 
                'read_all_partitions'] and isinstance(num_partitions, int) and num_partitions > 0
    elif isinstance(input_address, list):
        assert isinstance(table_name, list) and isinstance(schema, list) and \
               isinstance(read_pattern, list) and isinstance(num_partitions, list)
        assert len(input_address) == len(table_name) and len(input_address) == len(schema) and \
               len(input_address) == len(read_pattern) and len(input_address) == len(num_partitions)
        for i in range(len(input_address)):
            assert isinstance(input_address[i], str) and isinstance(table_name[i], str) and \
                   isinstance(schema[i], dict) and table_name[i] in schema[i] and \
                   isinstance(read_pattern[i], str) and read_pattern[i] in ['read_table', 
                   'read_partial_table', 'read_multiple_partitions', 'read_all_partitions'] and \
                   isinstance(num_partitions[i], int) and num_partitions[i] > 0
    else:
        assert False
    
    assert isinstance(output_address, str) and isinstance(storage_mode, str) and \
           storage_mode in ['s3', 'local']
    assert isinstance(suffix, str) and suffix in ['.dat', '.csv']
    assert func_id >= 0 and isinstance(func_id, int)

    key = {}
    key['task_id'] = task_id
    key['num_tasks'] = num_tasks

    key['input_address'] = input_address
    if isinstance(input_address, str):
        key['column_names'] = schema[table_name].keys()
        key['dtypes'] = {}
        for i in key['column_names']:
            key['dtypes'][i] = get_pd_type(schema[table_name][i])
    else:
        key['column_names'] = [schema[i][table_name[i]].keys() for i in range(len(table_name))]
        key['dtypes'] = [{} for i in table_name]
        for i in range(len(table_name)):
            for j in key['column_names'][i]:
                key['dtypes'][i][j] = get_pd_type(schema[i][table_name[i]][j])
    
    key['read_pattern'] = read_pattern
    key['num_partitions'] = num_partitions
    key['output_address'] = output_address
    key['storage_mode'] = storage_mode  
    key['suffix'] = suffix
    key['func_id'] = func_id
    return key


''' These functions are not used in the current version

def write_local_partitions(df, column_names, partitions, fn):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(column_names, list)
    assert isinstance(partitions, int)
    assert isinstance(fn, str)

    column_indices = [df.columns.get_loc(myterm) for myterm in column_names]

    row_count = int(df.shape[0])
    row_indices = np.array_split(np.arange(row_count), partitions)

    outputs_info = []
    for index, row_ind in enumerate(row_indices):
        df_part = df.iloc[row_ind, column_indices]
        output_loc = fn + '_' + str(index) + ".csv"
        outputs_info.append(write_local_intermediate(df_part, output_loc))
    
    results = {}
    results['outputs_info'] = outputs_info
    return results


def write_s3_partitions(df, column_names, partitions, fn, threadpool = True):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(column_names, list)
    assert isinstance(partitions, int)
    assert isinstance(fn, str)

    column_indices = [df.columns.get_loc(myterm) for myterm in column_names]

    row_count = int(df.shape[0])
    row_indices = np.array_split(np.arange(row_count), partitions)
    outputs_info = [i for i in range(partitions)]

    if threadpool:
        def write_task(index):
            row_ind = row_indices[index]
            df_part = df.iloc[row_ind, column_indices]
            output_loc = fn + '_' + str(index) + ".csv"
            outputs_info[index] = write_s3_intermediate(df_part, output_loc)

        write_pool = ThreadPool(1)
        write_pool.map(write_task, range(partitions))
        write_pool.close()
        write_pool.join()
    else:
        for index, row_ind in enumerate(row_indices):
            df_part = df.iloc[row_ind, column_indices]
            output_loc = fn + '_' + str(index) + ".csv"
            outputs_info[index] = write_s3_intermediate(df_part, output_loc)
    
    results = {}
    results['outputs_info'] = outputs_info
    return results


def write_partitions(df, column_names, bintype, partitions, fn, storage_mode = 's3'):
    if storage_mode == "local":
        return write_local_partitions(df, column_names, bintype, partitions, fn)
    elif storage_mode == "s3":
        return write_s3_partitions(df, column_names, bintype, partitions, fn)
    else:
        raise Exception("Invalid storage mode")


def read_intermediate(key, storage_mode = 's3'):
    if storage_mode == "local":
        return read_local_intermediate(key)
    elif storage_mode == "s3":
        return read_s3_intermediate(key)
    else:
        raise Exception("Invalid storage mode")


def read_local_multiple_splits(names, dtypes, prefix, number_splits, suffix):
    key = {}
    key['column_names'] = names
    dtypes_dict = {}
    for i in range(len(names)):
        dtypes_dict[names[i]] = dtypes[i]
    key['dtypes'] = dtypes_dict
    ds = []
    for i in range(number_splits):
        key['input_address'] = prefix + str(i) + suffix
        d = read_local_intermediate(key)
        ds.append(d)
    return pd.concat(ds)


def read_s3_multiple_splits(names, dtypes, prefix, number_splits, suffix, threadpool=True):
    dtypes_dict = {}
    for i in range(len(names)):
        dtypes_dict[names[i]] = dtypes[i]
        
    ds = [i for i in range(number_splits)]
    s3_client = boto3.client("s3")

    if threadpool:
        def read_work(split_index):
            key = {}
            key['column_names'] = names
            key['dtypes'] = dtypes_dict
            key['input_address'] = prefix + str(split_index) + suffix
            d = read_s3_intermediate(key, s3_client)
            ds[split_index] = d
        
        read_pool = ThreadPool(1)
        read_pool.map(read_work, range(number_splits))
        read_pool.close()
        read_pool.join()
    else:
        for i in range(number_splits):
            key = {}
            key['column_names'] = names
            key['dtypes'] = dtypes_dict
            key['input_address'] = prefix + str(i) + suffix
            d = read_s3_intermediate(key, s3_client)
            ds[i] = d

    return pd.concat(ds)


def read_multiple_splits(names, dtypes, prefix, number_splits, suffix, storage_mode = 's3'):
    if storage_mode == "local":
        return read_local_multiple_splits(names, dtypes, prefix, number_splits, suffix)
    elif storage_mode == "s3":
        return read_s3_multiple_splits(names, dtypes, prefix, number_splits, suffix)
    else:
        raise Exception("Invalid storage mode")
'''


# if __name__ == '__main__':
#     print(np.dtype("float32"))
#     key = {}
#     key['loc'] = 'tpcds/customer_demographics.dat'
#     key['names'] = table_schema['customer_demographics'].keys()

#     key['dtypes'] = {}
#     for i in key['names']:
#         key['dtypes'][i] = get_pd_type(table_schema['customer_demographics'][i])


#     part_data = read_s3_table(key)
#     print('---')

#     print(part_data.info())

#     print('---')

#     t1 = time.time()
#     res = write_s3_partitions(part_data, part_data.columns.tolist(), 10, 'tpcds/test/')
#     t2 = time.time()
#     print(t2 - t1)
#     print(res)