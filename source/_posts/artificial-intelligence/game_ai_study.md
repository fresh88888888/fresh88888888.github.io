---
title: 游戏AI & 强化学习
date: 2024-03-27 09:10:32
tags:
  - AI
categories:
  - 人工智能
---
#### 游戏AI

##### 环境设置

游戏环境配备了已经为您实现的代理。要查看这些默认代理的列表，请运行：
```python
from kaggle_environments import make, evaluate

# Create the game environment
# Set debug=True to see the errors if your agent refuses to run
env = make("connectx", debug=True)

# List of available default agents
print(list(env.agents))
# ['random', 'negamax']
```
<!-- more -->
“`random`”代理从有效移动集中（统一）随机选择。在四子棋中，如果该列中仍有空间放置棋子（即，如果棋盘有七行，则该列中的棋子少于七个），则移动被视为有效。在下面的代码单元中，该代理与自身的副本玩一轮游戏。
```python
# Two random agents play one game round
env.run(["random", "random"])

# Show the game
env.render(mode="ipython")
```
##### 定义代理

您将创建自己的代理。您的代理应实现为接受两个参数的`Python`函数：`obs`和`config`。它返回一个包含所选列的整数，其中索引从零开始。因此，返回值是`0-6`之一（含）。我们将从几个示例开始，以提供一些背景信息。在下面的代码单元中：
- 第一个代理的行为与上面的“随机”代理相同。
- 第二个代理总是选择中间的列，无论它是否有效！请注意，如果任何智能体选择了无效的移动，它就会输掉比赛。
- 第三个代理选择最左边的有效列。
```python
import random
import numpy as np

# Selects random valid column
def agent_random(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)

# Selects middle column
def agent_middle(obs, config):
    return config.columns//2

# Selects leftmost valid column
def agent_leftmost(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return valid_moves[0]
```
那么，`obs`和`config`到底是什么？`obs`包含两条信息：
- `obs.board`-游戏板（一个`Python`列表，每个网格位置有一个`item`）。
- `obs.mark`-分配给代理的标记（`1`或`2`）。

`obs.board`是一个显示棋子位置的`Python`列表，其中第一行首先出现，然后是第二行，依此类推。我们使用`1`来跟踪`1`的棋子，使用`2`来跟踪`2`的棋子。
{% asset_img ga_1.png %}

`obs.board`将为[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 2, 0, 2, 0]。

`config`包含三部分信息：
- `config.columns` - 游戏板中的列数（四子棋为`7`）
- `config.rows` - 游戏板的行数（四子棋为`6`）
- `config.inarow` - 玩家需要连续获得的棋子数量才能获胜（四子棋为`4`）

现在花点时间研究一下我们上面定义的三个代理。

##### 评估代理

为了让自定义代理玩一轮游戏，我们使用与之前相同的`env.run()`方法。
```python
# Agents play one game round
env.run([agent_leftmost, agent_random])

# Show the game
env.render(mode="ipython")
```
单场比赛的结果通常不足以说明我们的智能体表现如何。为了获得更好的想法，我们将计算每个代理在多场比赛中的平均获胜百分比。为了公平起见，每个代理人都先去除一半的时间。为此，我们将使用 `get_win_percentages()`函数。
```python
def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
```
您认为哪个代理在对抗随机代理时表现更好：总是在中间的代理(`agent_middle`)，还是选择最左边有效列的代理(`agent_leftmost`)？
```python
get_win_percentages(agent1=agent_middle, agent2=agent_random)
```
看起来选择最左边有效列的代理表现最好！

#### One-Step Lookahead

即使您是四子棋新手，您也可能已经制定了几种游戏策略。

##### 游戏树（Game trees）

作为一名人类玩家，你如何看待这个游戏的玩法？您可能会做一些预测。对于每个潜在的举动，您预测对手可能会做什么反应，以及您随后将如何反应，以及对手随后可能会做什么等等。然后，您选择您认为最有可能获胜的举动。我们可以形式化这个想法，并在**博弈树**中表示所有可能的结果。
{% asset_img ga_2.png %}

**游戏树**代表每个可能的动作（由代理和对手），从空棋盘开始。第一行显示代理（红色玩家）可以做出的所有可能的动作。接下来，我们记录对手（黄色玩家）可以做出的每一步反应，依此类推，直到每个分支到达游戏结束。（《四子棋》的游戏树非常大，因此我们在上图中仅显示了一个小预览。）一旦我们能够看到游戏可能结束的所有方式，它就可以帮助我们选择最有可能获胜的棋步。

##### 启发式（Heuristics）

四子棋的完整游戏树有超过`4`万亿个不同的棋盘！因此，在实践中，我们的代理在计划移动时仅使用一小部分。为了确保不完整的树对代理仍然有用，我们将使用**启发式**（或**启发式函数**）。 启发式将分数分配给不同的游戏板，我们估计得分较高的板更有可能导致代理赢得游戏。您将根据您对游戏的了解来设计启发式。例如，对于“四子连接”来说，一种可能相当有效的启发式方法会查看一条（水平、垂直或对角线）线上的每组四个相邻位置，并分配：
- 如果代理人连续有四个棋子（代理人获胜），则`1000000(1e6)`分。
- 如果代理人填补了三个位置，并且剩余位置为空，则得`1`分（如果代理人填补了空位，则代理人获胜）。
- 如果对手填满了三个位置，而剩下的位置是空的，则`-100`分（对手填满空位即获胜）。

{% asset_img ga_3.png %}

代理究竟将如何使用启发式？考虑轮到代理了，它正在尝试为下图顶部所示的游戏板计划一个动作。有七种可能的移动（每列一种）。对于每一步，我们都会记录最终的游戏板。
{% asset_img ga_4.png %}

然后我们使用启发式为每个板分配分数。为此，我们搜索网格并在启发式中查找该模式的所有出现，类似于单词搜索谜题。每次出现都会修改分数。例如，
- 第一个棋盘（智能体在第`0`列中进行游戏）得分为`2`。这是因为该棋盘包含两个不同的模式，每个模式都会为分数添加一分（两者都在上图中圈出）。
- 第二块板的得分为`1`。
- 第三块板（代理在第`2`列中进行游戏）得分为`0`。这是因为启发式中的任何模式都没有出现在板中。

第一个棋盘得分最高，因此智能体将选择此棋步。这对于玩家来说也是最好的结果，因为只要再走一步，它就可以保证获胜。现在检查一下图中的内容，以确保它对您有意义！对于这个特定的例子，启发式方法非常有效，因为它匹配了得分最高的最佳动作。这只是创建`Connect Four`代理的众多启发式方法之一，您可能会发现您可以设计一种效果更好的启发式方法！一般来说，如果您不确定如何设计启发式（即如何对不同的游戏状态进行评分，或者将哪些分数分配给不同的条件），通常最好的办法就是简单地进行初步猜测，然后进行游戏, 反对你的代理人。这将使您能够识别代理做出错误动作时的特定情况，然后您可以通过修改启发式来修复这些情况。

##### Code

使用启发式为每个可能的有效动作分配分数，并且选择得分最高的动作。（如果多个动作获得高分，我们随机选择一个。）“`One-Step Lookahead`”是指智能体仅展望未来的一步，而不是深入博弈树。为了定义这个代理，我们将使用下面代码单元中的函数。当我们使用这些函数来指定代理时，它们会更有意义。
```python
import random
import numpy as np

# Calculates score if agent drops piece in selected column
def score_move(grid, col, mark, config):
    next_grid = drop_piece(grid, col, mark, config)
    score = get_heuristic(next_grid, mark, config)
    return score

# Helper function for score_move: gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid

# Helper function for score_move: calculates value of heuristic for grid
def get_heuristic(grid, mark, config):
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_threes_opp = count_windows(grid, 3, mark%2+1, config)
    score = num_threes - 1e2*num_threes_opp + 1e6*num_fours
    return score

# Helper function for get_heuristic: checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)
    
# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def count_windows(grid, num_discs, piece, config):
    num_windows = 0
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows

# The agent is always implemented as a Python function that accepts two arguments: obs and config
def agent(obs, config):
    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next turn
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)
```
在代理的代码中，我们首先获取有效移动的列表。接下来，我们将游戏板转换为`2D numpy`数组。对于四子棋来说，网格是一个`6`行`7`列的数组。然后，`score_move()`函数计算每个有效移动的启发式值。它使用几个辅助函数：`drop_piece()`返回将其棋子放入所选列时生成的网格。`get_heuristic()`计算提供的板（网格）的启发值，其中`mark`是代理的标记。此函数使用 `count_windows()`函数，该函数根据启发式计算满足特定条件的窗口（行、列或对角线中的四个相邻位置）的数量。具体来说，`count_windows（grid，num_discs，piece，config）`产生游戏板（网格）中包含带有标记棋子的玩家（代理或对手）的`num_discs`棋子的窗口数量，并且窗口中的其余位置为空 。例如，设置`num_discs=4`和`piece=obs.mark`计算代理连续获得四个棋子的次数。设置`num_discs=3`和`piece=obs.mark%2+1`则统计对手有3个棋子且剩余位置为空的窗口数量（对手将空位填满即获胜）。最后，我们得到最大化启发式的列列表，并随机选择一个。
```python
from kaggle_environments import make, evaluate

# Create the game environment
env = make("connectx")

# Two random agents play one game round
env.run([agent, "random"])

# Show the game
env.render(mode="ipython")
```
{% asset_img ga_5.png %}

```python
def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))

get_win_percentages(agent1=agent, agent2="random")
```
结果输出为：
```bash
Agent 1 Win Percentage: 1.0
Agent 2 Win Percentage: 0.0
Number of Invalid Plays by Agent 1: 0
Number of Invalid Plays by Agent 2: 0
```
该代理的性能比随机代理好得多！
