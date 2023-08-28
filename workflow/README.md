# Run Jolteon and other baseline schedulers

```
-w: workflow_name = <ml|tpcds|video>
-bt: bound_type = <latency|cost>
-bv: bound_value
```

## Preprocess

1. Profile
```
python3 -u scheduler.py -w ml -p 1 > tmp.log
python3 -u scheduler.py -w video -p 1 > tmp.log
python3 -u scheduler.py -w tpcds -p 1 > tmp.log
```

2. Train
```
python3 -u scheduler.py -w ml -t 1 > tmp1.log
python3 -u scheduler.py -w video -t 1 > tmp.log
python3 -u scheduler.py -w tpcds -t 1 > tmp.log
```

# Overall performance

## ML-Pipeline

python3 -u scheduler.py -w ml -bt latency -bv 24 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 26 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 50 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 70 > tmp.log

Latency bound
Run each command for 6 times and take the last 5 results to eliminate the cold start effect.
```
python3 -u scheduler.py -w ml -bt latency -bv 18 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 20 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 25 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 30 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 40 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 60 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 80 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 100 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 120 > tmp.log
```

Cost bound
Run each command for 6 times and take the last 5 results
```
python3 -u scheduler.py -w ml -bt cost -bv 700 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 800 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 1000 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 1600 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 2000 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 2500 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 3000 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 3200 > tmp.log
```

Ditto
```
python3 -u scheduler.py -w ml -s ditto -bt latency -tp 40 -nv 4 > tmp.log
python3 -u scheduler.py -w ml -s ditto -bt cost -tp 10 -nv 0.6 > tmp.log
```

Caerus
```
python3 -u scheduler.py -w ml -s caerus -tp 40 -nv 2.5 > tmp.log
```

Orion
```
python3 -u scheduler.py -w ml -s orion -bt latency -bv 25 -tp 40 > tmp.log
python3 -u scheduler.py -w ml -s orion -bt latency -bv 30 -tp 40 > tmp.log
python3 -u scheduler.py -w ml -s orion -bt latency -bv 40 -tp 40 > tmp.log
python3 -u scheduler.py -w ml -s orion -bt latency -bv 50 -tp 40 > tmp.log
python3 -u scheduler.py -w ml -s orion -bt latency -bv 70 -tp 40 > tmp.log
python3 -u scheduler.py -w ml -s orion -bt latency -bv 80 -tp 40 > tmp.log
```

## Video-Analytics

Overall
```
python3 -u scheduler.py -w video -bt latency -bv 9 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 12 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 20 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 30 > tmp.log
python3 -u scheduler.py -w video -bt cost -bv 920 > tmp.log
python3 -u scheduler.py -w video -bt cost -bv 850 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 80 > tmp.log
```

Latency bound

```
python3 -u scheduler.py -w video -bt latency -bv 9 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 12 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 20 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 30 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 50 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 65 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 80 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 100 > tmp.log
python3 -u scheduler.py -w video -bt latency -bv 120 > tmp.log
```

Cost bound
```
python3 -u scheduler.py -w video -bt cost -bv 850 > tmp.log
python3 -u scheduler.py -w video -bt cost -bv 920 > tmp.log
python3 -u scheduler.py -w video -bt cost -bv 1000 > tmp.log
python3 -u scheduler.py -w video -bt cost -bv 1300 > tmp.log
python3 -u scheduler.py -w video -bt cost -bv 1500 > tmp.log
python3 -u scheduler.py -w video -bt cost -bv 1800 > tmp.log
python3 -u scheduler.py -w video -bt cost -bv 2000 > tmp.log
python3 -u scheduler.py -w video -bt cost -bv 2500 > tmp.log
```

Ditto
```
python3 -u scheduler.py -w video -s ditto -bt latency -tp 128 -nv 5 > tmp.log
python3 -u scheduler.py -w video -s ditto -bt cost -tp 8 -nv 1 > tmp.log
```

Caerus
```
python3 -u scheduler.py -w video -s caerus -tp 10 -nv 5 > tmp.log
```

Orion
```
python3 -u scheduler.py -w video -s orion -bt latency -bv 25 -tp 20 > tmp.log
python3 -u scheduler.py -w video -s orion -bt latency -bv 30 -tp 20 > tmp.log
python3 -u scheduler.py -w video -s orion -bt latency -bv 40 -tp 20 > tmp.log
python3 -u scheduler.py -w video -s orion -bt latency -bv 60 -tp 20 > tmp.log
python3 -u scheduler.py -w video -s orion -bt latency -bv 80 -tp 20 > tmp.log
```

## TPC-DS

Overall
```
python3 -u scheduler.py -w tpcds -bt cost -bv 2200 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 23 > tmp.log
python3 -u scheduler.py -w tpcds -bt cost -bv 2000 > tmp.log
python3 -u scheduler.py -w tpcds -bt cost -bv 1800 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 30 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 65 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 75 > tmp.log
```

Latency bound
```
python3 -u scheduler.py -w tpcds -bt latency -bv 23 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 30 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 35 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 40 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 55 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 65 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 75 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 78 > tmp.log
python3 -u scheduler.py -w tpcds -bt latency -bv 80 > tmp.log
```

Cost bound
```
python3 -u scheduler.py -w tpcds -bt cost -bv 1700 > tmp.log
python3 -u scheduler.py -w tpcds -bt cost -bv 1800 > tmp.log
python3 -u scheduler.py -w tpcds -bt cost -bv 1900 > tmp.log
python3 -u scheduler.py -w tpcds -bt cost -bv 2000 > tmp.log
python3 -u scheduler.py -w tpcds -bt cost -bv 2100 > tmp.log
python3 -u scheduler.py -w tpcds -bt cost -bv 2200 > tmp.log
python3 -u scheduler.py -w tpcds -bt cost -bv 2300 > tmp.log
python3 -u scheduler.py -w tpcds -bt cost -bv 2400 > tmp.log
```