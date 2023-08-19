# Run Jolteon and other baseline schedulers

```
-w: workflow_name = <ml|tpcds|video>
-bt: bound_type = <latency|cost>
-bv: bound_value
```

## Jolteon

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

3. Run
```
python3 -u scheduler.py -w ml -bt latency -bv 80 > tmp.log
python3 -u scheduler.py -w tpcds -bt cost -bv 150 > tmp.log
```

## Orion

1. Run
```
python3 -u scheduler.py -w ml -s orion -bt latency -bv 40 > tmp.log
```

## Ditto

1. Run
```
python3 -u scheduler.py -w ml -s ditto -bt latency > tmp.log
```

## Caerus

1. Run
```
python3 -u scheduler.py -w ml -s caerus > tmp.log
```

# Run overall performance 

Latency bound
Run each command for 6 times and take the last 5 results to eliminate the cold start effect.
```
python3 -u scheduler.py -w ml -bt latency -bv 20 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 25 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 30 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 40 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 67 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 80 > tmp.log
python3 -u scheduler.py -w ml -bt latency -bv 500 > tmp.log

python3 -u scheduler.py -w ml -bt cost -bv 800 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 900 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 1000 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 1600 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 2000 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 3000 > tmp.log
python3 -u scheduler.py -w ml -bt cost -bv 5000 > tmp.log
```