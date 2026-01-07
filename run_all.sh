#!/bin/bash

# Clear old log
> output.log

# Step 1: Prepare Splits (Run once)
if [ ! -d "splits" ]; then
    echo "Generating data splits..." | tee -a output.log
    python -u prepare_splits.py >> output.log 2>&1
fi

# Step 2: Run Experiments Sequentially
# Adjust datasets and seeds as needed
# DATASETS=("breast_cancer" "telco" "adult" "credit")
DATASETS=("credit") # Example dataset, extend as needed
SEEDS=({0..9}) # Example seeds, extend to {0..9} for full experiment

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "----------------------------------------------------------------" | tee -a output.log
        echo "Processing $dataset - Seed $seed" | tee -a output.log
        echo "----------------------------------------------------------------" | tee -a output.log
        
        # Run Holdout
        echo "[1/2] Running Holdout..." | tee -a output.log
        python -u run_experiment.py --dataset "$dataset" --seed "$seed" --task holdout >> output.log 2>&1
        
        # Run Nested CV
        echo "[2/3] Running Nested CV..." | tee -a output.log
        python -u run_experiment.py --dataset "$dataset" --seed "$seed" --task nested >> output.log 2>&1

        # Run Standard CV
        echo "[3/3] Running Standard CV..." | tee -a output.log
        python -u run_experiment.py --dataset "$dataset" --seed "$seed" --task standard_cv >> output.log 2>&1
        
        echo "Completed $dataset - Seed $seed" | tee -a output.log
    done
done

echo "All experiments completed." | tee -a output.log
