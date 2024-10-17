import numpy as np
import random
import json

import math

import numpy as np

agents_skills_custom_duration = {
    "1": {"1": 16},
    "2": {"1": 8},
    "3": {"2": 10},
    "4": {"2": 5},
    "5": {"7": 13}
}

new_agents_skills_custom_duration = {
    int(outer_key): {
        int(inner_key): value 
        for inner_key, value in inner_dict.items()
    } 
    for outer_key, inner_dict in agents_skills_custom_duration.items()
}


print(new_agents_skills_custom_duration)

current_agent = 5
action = 8

if current_agent in new_agents_skills_custom_duration and action in new_agents_skills_custom_duration[current_agent]:
    action_time = new_agents_skills_custom_duration[current_agent][action]
else:
    action_time = 10000

print(action_time)


