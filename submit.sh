
# run the sbatch <script> cmd for all scripts in `scripts/` folder
for script in ./scripts/*.sh; do
    echo "Submitting $script"
    sbatch "$script"
done