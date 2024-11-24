#!/bin/bash

# Remove previous log files if they exist
rm -f output_environments.log output_clients.log

sleep 5

# Run the servers and environment script, logging output
./run_servers_and_environment.sh >> output_environments.log

sleep 90

# Run the clients script, logging output
./run_clients.sh >> output_clients.log