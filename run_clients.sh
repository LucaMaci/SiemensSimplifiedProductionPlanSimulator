
n_agents=3
base_port=9980
base_seed=1000
scp=ai_optimizer/configs/simulator_config_"$n_agents"units.json
pcp=ai_optimizer/configs/simulator_config_"$n_agents"units.json
lcp=ai_optimizer/configs/learning_config.json

for ((i=0;i<$n_agents;i+=1))
do
      python3 ai_optimizer/cppu_path_learner_decentralized.py --cppu_name cppu_$i --learning_config_path $lcp --simulator_config_path $scp --product_config_path $pcp --seed $((base_seed + i)) --out_dir output/ --serverport $((base_port + i)) &
done