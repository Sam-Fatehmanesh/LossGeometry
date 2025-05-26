./run_analysis.py \
  --dataset mnist \
  --model mlp \
  --output_size 10 \
  --num_runs 64 \
  --num_epochs 10 \
  --learning_rate 0.01 \
  --momentum .9 \
  --analyze_singular_values \
  --experiment_name "mlp_mnist_momentum=.9" 