# Naural network based chess position evaluator

Trained on lichess evaluation dataset at https://database.lichess.org/#evals.
Uses optuna for hyperparameter optimization.

Used in a one depth chess engine and compared against:
- an engine making random moves
- an engine taking choosing move based on the material difference looking one move ahead
- stockfish 16 with a depth search of 1

### Results
After 20 epochs of training on a milion white side positions.
Created engine is named `Neural`. `Neural + Material` is a hybrid engine that uses both neural network prediction and material difference for deciding on a move.
The created engine plays better as black because it was trained to be better at evaluating white's position.

| Matchup                        | White wins | Black wins | Draws |
|--------------------------------|------------|------------|-------|
| Random vs Neural               | 0          | 27         | 3     |
| Random vs Neural + Material    | 0          | 25         | 5     |
| Material vs Neural             | 14         | 9          | 7     |
| Material vs Neural + Material  | 3          | 11         | 16    |
| Neural vs Neural               | 2          | 15         | 13    |
| Stockfish vs Neural            | 28         | 2          | 0     |
| Stockfish vs Neural + Material | 28         | 1          | 1     |
