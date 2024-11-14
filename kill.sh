#!/bin/bash

# Find the process ID (PID) of the process named "dummy_workload"
PID=$(ps -a | grep 'dynamic_workloa' | awk '{print $1}')

# Check if PID is found
if [ -n "$PID" ]; then
  # Kill the process
  kill $PID
  echo "Process dynamic_workload (PID: $PID) has been killed."
else
  echo "No process named dummy_workload found."
fi

