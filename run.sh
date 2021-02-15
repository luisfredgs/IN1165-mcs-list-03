#!/bin/sh

# KC2 

## OLA
python main.py --dynamic-selection --dynamic-algorithm=ola --dataset=kc2 --hardness=all_instances
python main.py --dynamic-selection --dynamic-algorithm=ola --dataset=kc2 --hardness=easy
python main.py --dynamic-selection --dynamic-algorithm=ola --dataset=kc2 --hardness=hard

## LCA
python main.py --dynamic-selection --dynamic-algorithm=lca --dataset=kc2 --hardness=all_instances
python main.py --dynamic-selection --dynamic-algorithm=lca --dataset=kc2 --hardness=easy
python main.py --dynamic-selection --dynamic-algorithm=lca --dataset=kc2 --hardness=hard

## MCB
python main.py --dynamic-selection --dynamic-algorithm=mcb --dataset=kc2 --hardness=all_instances
python main.py --dynamic-selection --dynamic-algorithm=mcb --dataset=kc2 --hardness=easy
python main.py --dynamic-selection --dynamic-algorithm=mcb --dataset=kc2 --hardness=hard

## KNORA-U
python main.py --dynamic-selection --dynamic-algorithm=knorau --dataset=kc2 --hardness=all_instances
python main.py --dynamic-selection --dynamic-algorithm=knorau --dataset=kc2 --hardness=easy
python main.py --dynamic-selection --dynamic-algorithm=knorau --dataset=kc2 --hardness=hard

## KNORA-E
python main.py --dynamic-selection --dynamic-algorithm=kne --dataset=kc2 --hardness=all_instances
python main.py --dynamic-selection --dynamic-algorithm=kne --dataset=kc2 --hardness=easy
python main.py --dynamic-selection --dynamic-algorithm=kne --dataset=kc2 --hardness=hard


# jm1

## OLA
python main.py --dynamic-selection --dynamic-algorithm=ola --dataset=jm1 --hardness=all_instances
python main.py --dynamic-selection --dynamic-algorithm=ola --dataset=jm1 --hardness=easy
python main.py --dynamic-selection --dynamic-algorithm=ola --dataset=jm1 --hardness=hard

## LCA
python main.py --dynamic-selection --dynamic-algorithm=lca --dataset=jm1 --hardness=all_instances
python main.py --dynamic-selection --dynamic-algorithm=lca --dataset=jm1 --hardness=easy
python main.py --dynamic-selection --dynamic-algorithm=lca --dataset=jm1 --hardness=hard

## MCB
python main.py --dynamic-selection --dynamic-algorithm=mcb --dataset=jm1 --hardness=all_instances
python main.py --dynamic-selection --dynamic-algorithm=mcb --dataset=jm1 --hardness=easy
python main.py --dynamic-selection --dynamic-algorithm=mcb --dataset=jm1 --hardness=hard

## KNORA-U
python main.py --dynamic-selection --dynamic-algorithm=knorau --dataset=jm1 --hardness=all_instances
python main.py --dynamic-selection --dynamic-algorithm=knorau --dataset=jm1 --hardness=easy
python main.py --dynamic-selection --dynamic-algorithm=knorau --dataset=jm1 --hardness=hard

## KNORA-E
python main.py --dynamic-selection --dynamic-algorithm=kne --dataset=jm1 --hardness=all_instances
python main.py --dynamic-selection --dynamic-algorithm=kne --dataset=jm1 --hardness=easy
python main.py --dynamic-selection --dynamic-algorithm=kne --dataset=jm1 --hardness=hard



python main.py --dataset=jm1 --hardness=all_instances
python main.py --dataset=jm1 --hardness=easy
python main.py --dataset=jm1 --hardness=hard

python main.py --dataset=kc2 --hardness=all_instances
python main.py --dataset=kc2 --hardness=easy
python main.py --dataset=kc2 --hardness=hard