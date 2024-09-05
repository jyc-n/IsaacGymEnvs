import sys
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

def combined_reward_plots(path, args):
  data = []
  for f in args:
    filename = Path(path) / f
    assert filename.is_file()

    file = pd.read_csv(filename)
    data.append(file)

  result = pd.concat(data, axis=0)
  result.sort_values(by=['Step'], inplace=True)
  result.drop_duplicates(subset=['Step'], keep='first', inplace=True)
  print(result)

  max_value = result['Value'].max()

  result.plot(x='Step', y='Value', lw=0.5)
  plt.xlabel('iter')
  plt.ylabel('reward')
  plt.hlines(max_value, 0, result['Step'].max(), color='k', ls='--')
  # plt.show()
  plt.savefig('reward_plot.png', dpi=300)

  return

if __name__ == '__main__':
  combined_reward_plots(sys.argv[1], sys.argv[2:])