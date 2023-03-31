#!/bin/bash

tmux new-session -d -s langchain-contrib 'cd docs && poetry run /usr/bin/make livehtml'
tmux split-window -v 'poetry run jupyter-lab'
tmux split-window -v 'poetry shell'
tmux attach-session -t langchain-contrib