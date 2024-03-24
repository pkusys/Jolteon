## 1. Introduction<br>
This repository contains one version of the source code for our NSDI'24 paper "Jolteon: Unleashing the Promise of Serverless for Serverless Workflows" [[Paper]](https://www.usenix.org/conference/nsdi24/presentation/zhang-zili-jolteon).


## 2. Content<br>

- ML-Pipeline/, Video-Analytics/ and TPC-DS/<br>
    - The source code of three workflows used in the evaluation.
- profiles/<br>
    - The profile results of the three workflows, which are based on AWS lambda.
- workflow/<br>
    - The source code of Jolteon, which contains the execution engine, performance model and
    scheduling algorithm.

## 3. Environment requirement<br>

- Hardware<br>
  - Register the function on AWS Lambda & Store data in AWS S3 & one c5.12xlarge EC2 instance for depolying Jolteon <br>
- Software<br>
  - Please refer to `repo/requirements.txt` and `repo/aws_setup.sh`.<br>

## 4. How to run<br>

Please refer to `repo/workflow/README.md` for details.

## 5. Contact<br>

For any question, please contact  `zzlcs at pku dot edu dot cn` or `chaojin at pku dot edu dot cn`.
