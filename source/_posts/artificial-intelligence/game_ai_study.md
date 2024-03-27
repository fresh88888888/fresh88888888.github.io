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

`obs.board`将为`[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 2, 0, 2, 0]`。

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

#### N-Step Lookahead

##### 介绍

您学习了如何构建`One-Step Lookahead`的代理。该代理表现相当不错，但绝对还有改进的空间！例如，考虑下图中的潜在走势。（请注意，我们对列使用从零开始的编号，因此最左边的列对应于 `col=0`，下一列对应于`col=1`，依此类推。）
{% asset_img ga_6.png %}

通过`One-Step Lookahead`，红色玩家选择第`5`列或第`6`列之一，每一列都有`50%`的概率。但是，第`5`列显然是一个糟糕的棋步，因为它让对手只需要多一个回合就能赢得比赛。不幸的是，智能体不知道这一点，因为它只能展望未来的一步。接下来，您将使用**极小极大算法**来帮助代理更长远地展望未来并做出更明智的决策。

##### Minimax算法

我们希望利用**游戏树**更深处的信息。现在，假设我们的深度为`3`。这样，在决定其移动时，代理会考虑所有可能的游戏板，这些游戏板可以由
- 代理人的移动。
- 对手的动作。
- 代理人的下一步行动。

我们将使用一个视觉示例。为简单起见，我们假设在每一回合，代理和对手都只有两种可能的动作。下图中的每个蓝色矩形对应着不同的游戏板。
{% asset_img ga_7.png %}

我们用启发式的分数标记了树底部的每个“叶节点”。和以前一样，当前的游戏板位于图的顶部，代理的目标是结束获得尽可能高的分数。但请注意，智能体不再完全控制其分数——在智能体采取行动后，对手选择自己的行动。而且，对手的选择对于玩家来说可能是灾难性的！尤其，
- 如果智能体选择左边的分支，对手可以强制得分为`-1`。
- 如果智能体选择了正确的分支，对手可以强制得分`+10`。

现在花点时间检查一下图中的这一点，以确保它对您有意义！考虑到这一点，您可能会认为正确的分支对于代理来说是更好的选择，因为它是风险较小的选择。当然，它放弃了获得只能在左侧分支上访问的大分数（`+40`）的可能性，但它也保证了代理至少获得`+10`分。这是**极小极大算法**背后的主要思想：**智能体选择移动以获得尽可能高的分数，并且假设对手将通过选择移动来迫使分数尽可能低来抵消这一点**。也就是说，智能体和对手有相反的目标，我们假设对手发挥最佳。那么，在实践中，智能体如何利用这个假设来选择行动呢？我们在下图中说明了智能体的思维过程。
{% asset_img ga_8.png %}

在该示例中，`minimax`为左侧的移动分配`-1`分，为右侧的移动分配`+10`分。因此，智能体将选择右侧的移动。

##### Code

```python
import random
import numpy as np

# Gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid

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
```
我们还需要稍微修改一下**启发式**，因为对手现在能够修改游戏板。
{% asset_img ga_9.png %}

我们需要通过下棋来检查对手是否赢得了比赛。新的启发式方法查看（水平、垂直或对角线）线上的每组四个相邻位置并分配：
- 如果代理人连续有四张棋子（代理人获胜），则`1000000`(`1e6`) 分。
- 如果代理人填补了三个位置，并且剩余位置为空，则得`1`分（如果代理人填补了空位，则代理人获胜）。
- 如果对手填满了三个位置，而剩余位置为空（对手填满空位则获胜），则`-100`分。
- 如果对手连续有四张棋子（对手获胜），则`-10000`(`-1e4`) 分。

```python
# Helper function for minimax: calculates value of heuristic for grid
def get_heuristic(grid, mark, config):
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_threes_opp = count_windows(grid, 3, mark%2+1, config)
    num_fours_opp = count_windows(grid, 4, mark%2+1, config)
    score = num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours
    return score
```
我们定义了**极小极大代理**所需的一些附加函数。
```python
# Uses minimax to calculate value of dropping piece in selected column
def score_move(grid, col, mark, config, nsteps):
    next_grid = drop_piece(grid, col, mark, config)
    score = minimax(next_grid, nsteps-1, False, mark, config)
    return score

# Helper function for minimax: checks if agent or opponent has four in a row in the window
def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(2) == config.inarow

# Helper function for minimax: checks if game has ended
def is_terminal_node(grid, config):
    # Check for draw 
    if list(grid[0, :]).count(0) == 0:
        return True
    # Check for win: horizontal, vertical, or diagonal
    # horizontal 
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if is_terminal_window(window, config):
                return True
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if is_terminal_window(window, config):
                return True
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    return False

# Minimax implementation
def minimax(node, depth, maximizingPlayer, mark, config):
    is_terminal = is_terminal_node(node, config)
    valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
    if depth == 0 or is_terminal:
        return get_heuristic(node, mark, config)
    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark, config)
            value = max(value, minimax(child, depth-1, False, mark, config))
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark%2+1, config)
            value = min(value, minimax(child, depth-1, True, mark, config))
        return value
```
`N_STEPS`变量用于设置树的深度。
```python
# How deep to make the game tree: higher values take longer to run!
N_STEPS = 3

def agent(obs, config):
    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)
```
我们看到与随机代理的一轮游戏的结果。
```python
from kaggle_environments import make, evaluate

# Create the game environment
env = make("connectx")

# Two random agents play one game round
env.run([agent, "random"])

# Show the game
env.render(mode="ipython")
```
我们会检查它的平均表现。
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

get_win_percentages(agent1=agent, agent2="random", n_rounds=50)
```
结果输出为：
```bash
Agent 1 Win Percentage: 1.0
Agent 2 Win Percentage: 0.0
Number of Invalid Plays by Agent 1: 0
Number of Invalid Plays by Agent 2: 0
```
#### 深度强化学习（Deep Reinforcement Learning）

##### 介绍

到目前为止，我们的玩家依赖于有关如何玩游戏的详细信息。**启发式**确实提供了很多关于如何选择动作的指导！接下来，您将学习如何使用**强化学习**来构建**智能代理**，而无需使用启发式方法。相反，我们将随着时间的推移逐渐完善代理的策略，只需玩游戏并尝试最大化获胜率。

##### 神经网络

很难想出一个完美的**启发式**。改进启发式通常需要多次玩游戏，以确定代理可以做出更好选择的特定情况。而且，要解释到底出了什么问题，并最终纠正旧错误而不意外引入新错误，可能具有挑战性。如果我们有更系统的方法来提升智能体的游戏体验，不是会容易很多吗？为了实现这一目标，我们将用**神经网络代替启发式方法**。网络接受当前板作为输入。并且，它输出每个可能的移动的概率。
{% asset_img ga_10.png %}

然后，代理通过从这些概率中采样来选择移动。例如，对于上图中的游戏板，智能体以`50%`的概率选择第`4`列。这样，为了编码一个好的游戏策略，我们只需要修改**网络的权重**，以便对于每个可能的游戏板，为更好的动作分配更高的概率。至少在理论上，这是我们的目标。在实践中，我们实际上不会检查，四子棋有超过`4`万亿个可能的游戏板！

##### 设置

在实践中，我们如何完成修改网络权重的任务？
- 每次移动后，我们都会给智能体一个奖励，告诉它做得有多好：
    - 如果智能体在该举动中赢得了游戏，我们将给予它`+1`的奖励。
    - 否则，如果智能体采取了无效的行动（结束了游戏），我们将给予它`-10`的奖励。
    - 否则，如果对手在下一步行动中赢得了比赛（即代理未能阻止对手获胜），我们将给予代理奖励`-1`。
    - 否则，代理将获得`1/42`的奖励。
- 每场比赛结束时，智能体都会将其奖励相加。我们将奖励的总和称为代理的累积奖励。
    - 例如，如果游戏持续`8`步（每个玩家玩四次），并且智能体最终获胜，则其累积奖励为`3*(1/42) + 1`。
    - 如果游戏持续`11`步（对手先走，因此智能体下棋五次），并且对手在最后一步获胜，则智能体的累积奖励为`4*(1/42) - 1`。
    - 如果游戏以平局结束，则智能体正好下完 21 步，并获得`21*(1/42)` 的累积奖励。
    - 如果游戏持续`7`步并以智能体选择无效的移动而结束，则智能体获得的累积奖励为`3*(1/42) - 10`。

我们的目标是找到（平均）最大化代理累积奖励的**神经网络权重**。这种使用**奖励**来跟踪代理表现的想法是强化学习领域的核心思想。一旦我们以这种方式定义问题，我们就可以使用各种**强化学习算法**中的任何一种来生成代理。

##### 强化学习（Reinforcement Learning）

强化学习算法有很多种，例如`DQN、A2C`和`PPO`等。所有这些算法都使用类似的过程来生成代理：
- 最初，权重设置为随机值。
- 当代理玩游戏时，算法会不断尝试新的权重值，以了解平均累积奖励受到的影响。随着时间的推移，在玩了很多游戏之后，我们很好地了解了权重如何影响**累积奖励**，并且算法会选择表现更好的权重。当然，我们在这里掩盖了细节，这个过程涉及很多复杂性。 现在，我们关注大局！
- 这样，我们最终会得到一个试图赢得游戏的代理（因此它获得`+1`的最终奖励，并避免`-1`和`-10`）并尝试使游戏持续尽可能长的时间（因此 它会尽可能多地收集`1/42`奖金）。您可能会争辩说，希望游戏持续尽可能长的时间并没有真正意义`-`这可能会导致代理效率非常低，在游戏早期不会采取明显的获胜动作。而且，你的直觉是正确的——这将使智能体需要更长的时间才能下出获胜的棋步！我们加入`1/42`奖励的原因是为了帮助我们将使用的算法更好地收敛。

##### Code

网上有很多强化学习算法的优秀实现。为了使环境与稳定基线兼容，我们需要做一些额外的工作。为此，我们定义了下面的`ConnectFourGym`类。此类将`ConnectX`实现为`OpenAI Gym`环境并使用多种方法：
- `Reset()`将在每个游戏开始时被调用。它以`6`行`7`列的`2D numpy`数组形式返回起始游戏板。
- `change_reward()`自定义代理收到的奖励。（比赛已经有自己的奖励系统，用于对代理进行排名，并且此方法会更改值以匹配我们设计的奖励系统。）
- `step()`用于播放代理的动作选择（作为动作提供）以及对手的响应。
    - 生成的游戏板（作为`numpy`数组）。
    - 代理的奖励（仅来自最近的移动：`+1、-10、-1`或`1/42`之一）。
    - 游戏是否结束（如果游戏结束，`done=True`否则，`done=False`）。

```python
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from kaggle_environments import make, evaluate
from gym import spaces

class ConnectFourGym(gym.Env):
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=2, 
                                            shape=(1,self.rows,self.columns), dtype=int)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None
    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(1,self.rows,self.columns)
    def change_reward(self, old_reward, done):
        if old_reward == 1: # The agent won the game
            return 1
        elif done: # The opponent won the game
            return -1
        else: # Reward 1/42
            return 1/(self.rows*self.columns)
    def step(self, action):
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid: # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else: # End the game and penalize agent
            reward, done, _ = -10, True, {}
        return np.array(self.obs['board']).reshape(1,self.rows,self.columns), reward, done, _
```
我们将训练一个代理来击败随机代理。我们在下面的`agent2`参数中指定这个对手。下一步是**指定神经网络的架构**。在本例中，我们使用**卷积神经网络**。
{% note warning %}
请注意，这是输出选择每一列的概率的神经网络。由于我们使用`PPO`算法，我们的网络还将输出一些附加信息（称为输入的“值”）。
{% endnote %}

```python
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Neural network for predicting action values
class CustomCNN(BaseFeaturesExtractor):
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
)
        
# Initialize agent
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)
```
在上面的代码单元中，神经网络的权重最初设置为随机值。在下一个代码单元中，我们“**训练代理**”，这只是我们找到可能导致代理选择良好动作的神经网络权重的另一种方式。
```python
# Train agent
model.learn(total_timesteps=60000)

def agent1(obs, config):
    # Use the best model to select a column
    col, _ = model.predict(np.array(obs['board']).reshape(1, 6,7))
    # Check if selected column is valid
    is_valid = (obs['board'][int(col)] == 0)
    # If not valid, select random move. 
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])
```
在下一个代码单元中，我们看到与随机代理的一轮游戏的结果。
```python
# Create the game environment
env = make("connectx")

# Two random agents play one game round
env.run([agent1, "random"])

# Show the game
env.render(mode="ipython")
```
并且，我们计算它相对于随机代理的平均表现。
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

get_win_percentages(agent1=agent1, agent2="random")
```
结果输出为：
```python
Agent 1 Win Percentage: 0.68
Agent 2 Win Percentage: 0.32
Number of Invalid Plays by Agent 1: 0
Number of Invalid Plays by Agent 2: 0
```
需要注意的是，我们在这里创建的代理只是经过训练来击败随机代理，因为它的所有游戏体验都是以随机代理为对手。如果我们想要产生一个比许多其他智能体可靠地表现更好的智能体，我们必须在训练期间将我们的智能体暴露给这些其他智能体。
