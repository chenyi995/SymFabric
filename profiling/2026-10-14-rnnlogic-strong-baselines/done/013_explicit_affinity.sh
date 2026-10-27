command -v slurmd >/dev/null && slurmd -C -vvvv > slurm_cpu_map.txt 2>&1
export OMP_PLACES='{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23}'
taskset -c 8-23 /home/cw636/venvs/epic/bin/python affinity_probe.py --output explicit_affinity.json
