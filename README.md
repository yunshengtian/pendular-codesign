# pendular-codesign

Final project of 6.832

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

