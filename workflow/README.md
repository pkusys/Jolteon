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
python3 -u scheduler.py -w tpcds -p 1 > tmp.log
python3 -u scheduler.py -w video -p 1 > tmp.log
```

2. Train
```
python3 -u scheduler.py -w ml -t 1 > tmp1.log
python3 -u scheduler.py -w tpcds -t 1 > tmp.log
python3 -u scheduler.py -w video -t 1 > tmp.log
```

# Run overall performance (ML-Pipeline)

Latency bound
Run each command for 6 times and take the last 5 results to eliminate the cold start effect.
```
python3 -u scheduler.py -w ml -bt latency -bv 20 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 25 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 30 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 40 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 80 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 500 > tmp.log
```

Cost bound
Run each command for 6 times and take the last 5 results
```
python3 -u scheduler.py -w ml -bt cost -bv 800 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 900 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 1000 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 1600 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 2000 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 3000 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 5000 > tmp.log
```

Ditto
Run each command for 3 times and take the last 5 results
```
python3 -u scheduler.py -w ml -s ditto -bt latency -tp 40 -nv 4 > tmp.log
python3 -u scheduler.py -w ml -s ditto -bt cost -tp 10 -nv 0.5 > tmp.log
```

Caerus
Run each command for 3 times and take the last 5 results
```
python3 -u scheduler.py -w ml -s caerus -tp 40 -nv 4 > tmp.log
python3 -u scheduler.py -w ml -s caerus -tp 10 -nv 0.5 > tmp.log
```

Orion
```
python3 -u scheduler.py -w ml -s orion -bt latency -bv 25 -tp 40 > tmp.log
python3 -u scheduler.py -w ml -s orion -bt latency -bv 25 -tp 40 -f 1 > tmp.log

python3 -u scheduler.py -w ml -s orion -bt latency -bv 30 -tp 40 > tmp.log
python3 -u scheduler.py -w ml -s orion -bt latency -bv 30 -tp 40 -f 1 > tmp.log

python3 -u scheduler.py -w ml -s orion -bt latency -bv 40 -tp 40 > tmp.log
python3 -u scheduler.py -w ml -s orion -bt latency -bv 40 -tp 40 -f 1 > tmp.log

python3 -u scheduler.py -w ml -s orion -bt latency -bv 50 -tp 40 > tmp.log
python3 -u scheduler.py -w ml -s orion -bt latency -bv 50 -tp 40 -f 1 > tmp.log

python3 -u scheduler.py -w ml -s orion -bt latency -bv 60 -tp 40 > tmp.log
python3 -u scheduler.py -w ml -s orion -bt latency -bv 60 -tp 40 -f 1 > tmp.log

python3 -u scheduler.py -w ml -s orion -bt latency -bv 80 -tp 40 > tmp.log
python3 -u scheduler.py -w ml -s orion -bt latency -bv 80 -tp 40 -f 1 > tmp.log
```