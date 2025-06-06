import wandb
import pandas as pd
from datetime import datetime

api = wandb.Api()
runs = api.runs("tiffany8-mit/TinyZero")
# print([run.name for run in runs])
print("shows wandb steps with time stamp for first and latest runs")

for run in (runs[0],runs[-1]):
	history = run.history(samples=500)  # adjust keys as needed
	print("history cols", history.columns)
	
	if "_timestamp" not in history.columns or "_step" not in history.columns: print('not found _timestamp or _step')
	history['timestamp'] = history['_timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
	print((history['_timestamp'].iloc[2] -  history['_timestamp'].iloc[1])/(history['_step'].iloc[2] -  history['_step'].iloc[1]))
	# The returned DataFrame is sorted in ascending order by default.
	print(run.name,run.state)
	print(history[['timestamp', '_step']].head(500))
