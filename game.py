import os
import chess
import chess.pgn
from typing import Literal
from stockfish import Stockfish
import torch
from helpers.fen import fen_to_bitboard
from helpers.neural_network import NeuralNetwork
from time import time


class NeuralEngine:
    def __init__(self):
        weights = torch.load("./models/200k_10e_2024_01_14.pt")
        self.model = NeuralNetwork(input_shape=[6, 8, 8]).to("cpu")
        self.model.load_state_dict(weights)
        self.model.eval()
        self.board = chess.Board()
        self.best_move = None
        self.best_evaluation = None

    def next_move(self, opponent_move: chess.Move) -> chess.Move:
        if opponent_move:
            self.board.push(opponent_move)

        self.best_move = None
        self.best_evaluation = None

        for move in self.board.legal_moves:
            self.board.push(move)
            evaluation = self.__evaluate(self.board.fen())
            if self.best_evaluation is None or self.best_evaluation < evaluation:
                self.best_evaluation = evaluation
                self.best_move = move
            self.board.pop()

        self.board.push(self.best_move)
        return self.best_move

    def reset(self):
        self.board = chess.Board()

    def __evaluate(self, position):
        X = fen_to_bitboard(position, merge_colors=True)
        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0)
        y = self.model(X).to("cpu")
        return y.detach().numpy()[0][0]


class StockfishEngine:
    # API: https://disservin.github.io/stockfish-docs/stockfish-wiki/UCI-&-Commands.html
    def __init__(self, elo: int):
        self.elo = elo
        self.stockfish = Stockfish(
            path="./engines/stockfish/stockfish-windows-x86-64-avx2.exe"
        )
        self.stockfish.set_elo_rating(elo)

    def next_move(self, opponent_move: chess.Move | None) -> chess.Move:
        if opponent_move:
            self.stockfish.make_moves_from_current_position([opponent_move.uci()])

        next_move = self.stockfish.get_best_move()
        self.stockfish.make_moves_from_current_position([next_move])

        return chess.Move.from_uci(next_move)

    def reset(self, start_position):
        self.stockfish.set_fen_position(start_position)


class Game:
    def __init__(
        self, stockfish_elo: int = 1320
    ):  # stockfish documentation says: type spin default 1320 min 1320 max 3190
        self.board = chess.Board()
        self.stockfish = StockfishEngine(stockfish_elo)
        self.custom = NeuralEngine()

    def play(self, starts: Literal["stockfish", "custom"]) -> chess.pgn.Game:
        turn = starts
        move = None
        while not (self.board.is_checkmate() or self.board.is_stalemate()):
            if turn == "stockfish":
                move = self.stockfish.next_move(move)
                turn = "custom"
            else:
                move = self.custom.next_move(move)
                turn = "stockfish"

            self.board.push(move)

        game = chess.pgn.Game.from_board(self.board)

        game.headers["Event"] = f"Pojedynek ze Stockfishem {self.stockfish.elo}"
        game.headers["Site"] = "Arena"

        if starts == "stockfish":
            game.headers["White"] = "Stockfish"
            game.headers["Black"] = "Custom"
        else:
            game.headers["White"] = "Custom"
            game.headers["Black"] = "Stockfish"

        return game, self.board.outcome(), len(self.board.move_stack)

    def reset(self):
        self.board = chess.Board()
        self.custom.reset()

        fen_start = self.board.fen()
        self.stockfish.reset(fen_start)

    @staticmethod
    def save(game: chess.pgn.Game, path: str):
        with open(path, "w") as pgn_file:
            pgn_file.write(str(game))


###
### EXAMPLE
###

if __name__ == "__main__":
    directory = "./games"
    try:
        os.mkdir(directory)
    except OSError as error:
        print(f"Directory {directory} already exists")

    game_simulator = Game(stockfish_elo=600)

    for i in range(10):
        t0 = time()
        p1, p2 = "custom", "stockfish"
        game, outcome, moves = game_simulator.play(p1)
        winner = p1 if outcome.winner == chess.WHITE else p2
        print(f"Game {i+1:2} took {time() - t0:.3}s - {winner} won in {moves} moves")

        file = f"game_{i+1}.pgn"
        Game.save(game, os.path.join(directory, file))

        game_simulator.reset()

# Upload game here to see game evaluation https://lichess.org/paste
