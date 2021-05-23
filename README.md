# pendular-codesign

Design and Control Co-Optimization of Pendular Robots (Final project of MIT 6.832 Underactuated Robotics)

## Installation

```
conda env create -f environment.yml
```

## Control Optimization

#### Pendulum

```
python control/test.py --env pendulum --control ilqr 
python control/test.py --env pendulum --control mppi
```

#### Acrobot

```
python control/test.py --env acrobot --control ilqr 
python control/test.py --env acrobot --control mppi
```

## Design Optimization

#### Pendulum

```
python optimize.py --env pendulum --control ilqr
python optimize.py --env pendulum --control mppi
```

#### Acrobot

```
python optimize.py --env acrobot --control ilqr
```


## Hyperparameters

To reproduce results in the report, please run

```
python optimize.py --env pendulum --control ilqr --num-iter 30 --lr 3e-2
python optimize.py --env pendulum --control mppi --num-iter 30 --lr 3e-2
python optimize.py --env acrobot --control ilqr --num-iter 50 --lr 1e-2
```

