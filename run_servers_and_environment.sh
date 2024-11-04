
n_agents=20
base_port=9980
scp=ai_optimizer/configs/simulator_config_siemens_"$n_agents"units_SAC.json
pcp=ai_optimizer/configs/simulator_config_siemens_"$n_agents"units_SAC.json
lcp=ai_optimizer/configs/learning_config.json

python3 run_environment.py &
for ((i=0;i<$n_agents;i+=1))
  do
      python3 ai_optimizer/server_creator.py --addr localhost --port $((base_port + i)) --scp $scp --product_config_path $pcp --lcp $lcp --workers 0 --cppu cppu_$i --algo SAC --checkpoint 0 --outdir output/ --seed $i &
done