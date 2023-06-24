-- begin query 1 in stream 0 using template query1.tpl
with customer_total_return as (
  select
    sr_customer_sk as ctr_customer_sk,
    sr_store_sk as ctr_store_sk,
    sum(sr_fee) as ctr_total_return
  from
    store_returns,
    date_dim
  where
    sr_returned_date_sk = d_date_sk
    and d_year = 2002
  group by
    sr_customer_sk,
    sr_store_sk
)
select
  c_customer_id
from
  customer_total_return ctr1,
  store,
  customer
where
  ctr1.ctr_total_return > (
    select
      avg(ctr_total_return) * 1.2
    from
      customer_total_return ctr2
    where
      ctr1.ctr_store_sk = ctr2.ctr_store_sk
  )
  and s_store_sk = ctr1.ctr_store_sk
  and s_state = 'MI'
  and ctr1.ctr_customer_sk = c_customer_sk
order by
  c_customer_id
limit
  100;
-- end query 1 in stream 0 using template query1.tpl

-- begin query 2 in stream 0 using template query16.tpl
select
  count(distinct cs_order_number) as "order count",
  sum(cs_ext_ship_cost) as "total shipping cost",
  sum(cs_net_profit) as "total net profit"
from
  catalog_sales cs1,
  date_dim,
  customer_address,
  call_center
where
  d_date between '2002-2-01'
  and (cast('2002-2-01' as date) + 60 days)
  and cs1.cs_ship_date_sk = d_date_sk
  and cs1.cs_ship_addr_sk = ca_address_sk
  and ca_state = 'NY'
  and cs1.cs_call_center_sk = cc_call_center_sk
  and cc_county in (
    'Ziebach County',
    'Levy County',
    'Huron County',
    'Franklin Parish',
    'Daviess County'
  )
  and exists (
    select
      *
    from
      catalog_sales cs2
    where
      cs1.cs_order_number = cs2.cs_order_number
      and cs1.cs_warehouse_sk <> cs2.cs_warehouse_sk
  )
  and not exists(
    select
      *
    from
      catalog_returns cr1
    where
      cs1.cs_order_number = cr1.cr_order_number
  )
order by
  count(distinct cs_order_number)
limit
  100;
-- end query 2 in stream 0 using template query16.tpl

-- begin query 3 in stream 0 using template query94.tpl
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
  d_date between '1999-2-01'
  and (cast('1999-2-01' as date) + 60 days)
  and ws1.ws_ship_date_sk = d_date_sk
  and ws1.ws_ship_addr_sk = ca_address_sk
  and ca_state = 'MD'
  and ws1.ws_web_site_sk = web_site_sk
  and web_company_name = 'pri'
  and exists (
    select
      *
    from
      web_sales ws2
    where
      ws1.ws_order_number = ws2.ws_order_number
      and ws1.ws_warehouse_sk <> ws2.ws_warehouse_sk
  )
  and not exists(
    select
      *
    from
      web_returns wr1
    where
      ws1.ws_order_number = wr1.wr_order_number
  )
order by
  count(distinct ws_order_number)
limit
  100;
-- end query 3 in stream 0 using template query94.tpl

-- begin query 4 in stream 0 using template query95.tpl
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
-- end query 4 in stream 0 using template query95.tpl
