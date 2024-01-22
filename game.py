import os
import chess
import chess.pgn
from typing import Literal
from stockfish import Stockfish
import torch
from helpers.fen import fen_to_bitboard
from helpers.neural_network import NeuralNetwork
from time import time
import numpy as np
from abc import ABC, abstractmethod
import datetime


class Engine(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def next_move(opponent_move: chess.Move):
        """Return the best move to the opponents move"""

    @abstractmethod
    def reset(fen: str):
        """Clear engine's inner state"""


class RandomEngine(Engine):
    def __init__(self):
        super().__init__("Random Engine")
        self.board = chess.Board()

    def next_move(self, opponent_move: chess.Move) -> chess.Move:
        if opponent_move:
            self.board.push(opponent_move)
        move = np.random.choice(list(self.board.legal_moves))
        self.board.push(move)
        return move

    def reset(self, fen: str):
        self.board = chess.Board(fen)


class NeuralEngine(Engine):
    def __init__(self, isWhite: bool, withMaterial=False):
        super().__init__("Neural Engine")
        self.isWhite = isWhite
        weights = torch.load("./models/mil_20e_dropout.pt")
        self.model = NeuralNetwork(input_shape=[12, 8, 8]).to("cpu")
        self.model.load_state_dict(weights)
        self.model.eval()
        self.board = chess.Board()
        self.withMaterial = withMaterial

    def next_move(self, opponent_move: chess.Move) -> chess.Move:
        if opponent_move:
            self.board.push(opponent_move)

        best_move = None
        best_evaluation = None

        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.is_checkmate():
                return move
            evaluation = self.__evaluate(self.board.fen())
            # evaluation += np.random.uniform(-0.1, 0.1)
            if self.withMaterial:
                evaluation += MaterialEngine.material_balance(self.board) / 15
            if not self.isWhite:
                evaluation *= -1
            if best_evaluation is None or best_evaluation < evaluation:
                best_evaluation = evaluation
                best_move = move
            self.board.pop()

        self.board.push(best_move)
        return best_move

    def reset(self, fen: str):
        self.board = chess.Board(fen)

    def __evaluate(self, fen):
        X = fen_to_bitboard(fen, merge_colors=False)
        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0)
        y = self.model(X).to("cpu")
        return y.detach().numpy()[0][0]


class MaterialEngine(Engine):
    def __init__(self, isWhite: bool):
        super().__init__("Material Engine")
        self.isWhite = isWhite
        self.board = chess.Board()

    def next_move(self, opponent_move: chess.Move) -> chess.Move:
        if opponent_move:
            self.board.push(opponent_move)

        best_evaluation = None
        best_move = None

        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.is_checkmate():
                return move
            evaluation = self.material_balance(self.board) + np.random.uniform(-1, 1)
            if not self.isWhite:
                evaluation *= -1
            if best_evaluation is None or best_evaluation < evaluation:
                best_evaluation = evaluation
                best_move = move
            self.board.pop()

        self.board.push(best_move)
        return best_move

    @staticmethod
    def material_balance(board):
        white = board.occupied_co[chess.WHITE]
        black = board.occupied_co[chess.BLACK]
        # fmt: off
        return (
            chess.popcount(white & board.pawns) - chess.popcount(black & board.pawns)
            + 3 * (chess.popcount(white & board.knights) - chess.popcount(black & board.knights))
            + 3 * (chess.popcount(white & board.bishops) - chess.popcount(black & board.bishops))
            + 5 * (chess.popcount(white & board.rooks) - chess.popcount(black & board.rooks))
            + 9 * (chess.popcount(white & board.queens) - chess.popcount(black & board.queens))
        # fmt: on
        )

    def reset(self, fen: str):
        self.board = chess.Board(fen)


class StockfishEngine(Engine):
    # API: https://disservin.github.io/stockfish-docs/stockfish-wiki/UCI-&-Commands.html
    def __init__(self, elo: int, depth: int = 15):
        super().__init__(f"Stockfish {elo} ({depth})")
        self.elo = elo
        self.stockfish = Stockfish(
            path=r"D:\\chess-ai\\engines\\stockfish\\stockfish-windows-x86-64-avx2.exe",
            depth=depth,
            parameters={"MultiPV": 10, "Skill Level": 0},
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
        self, engine1: Engine, engine2: Engine
    ):  # stockfish documentation says: type spin default 1320 min 1320 max 3190
        self.board = chess.Board()
        self.round = 0
        self.engines = [engine1, engine2]

    def play(self, starts: Literal[0, 1]) -> chess.pgn.Game:
        self.round += 1
        turn = starts
        move = None
        while not (self.board.is_checkmate() or self.board.is_stalemate()):
            if self.board.outcome():
                print("--- ", self.board.outcome())
                break
            current_engine = self.engines[0] if turn % 2 == starts else self.engines[1]
            move = current_engine.next_move(move)
            turn += 1
            self.board.push(move)

        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = f"{self.engines[0].name} vs {self.engines[1].name}"
        game.headers["Site"] = "Arena"
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = self.round
        game.headers["White"] = self.engines[starts].name
        game.headers["Black"] = self.engines[(starts + 1) % 2].name

        return game, self.board.outcome(), len(self.board.move_stack)

    def reset(self):
        self.board = chess.Board()
        self.engines[0].reset(chess.STARTING_FEN)
        self.engines[1].reset(chess.STARTING_FEN)

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

    # e1, e2 = RandomEngine(), NeuralEngine(False, True)
    # e1, e2 = MaterialEngine(True), NeuralEngine(False, False)
    # e1, e2 = NeuralEngine(True, False), NeuralEngine(False, False)
    e1, e2 = StockfishEngine(1320, 1), NeuralEngine(False, False)
    game_simulator = Game(e1, e2)
    scores = [0, 0]
    timeouts = 0

    for i in range(30):
        t0 = time()
        out = game_simulator.play(starts=0)
        if out:
            game, outcome, moves = out
            winner = "No one"
            if outcome is not None and outcome.winner is not None:
                winner = e1.name if outcome.winner == chess.WHITE else e2.name
                if outcome.winner in [chess.WHITE, chess.BLACK]:
                    scores[int(outcome.winner == chess.BLACK)] += 1
                    print(
                        f"Game {i+1:2} took {time() - t0:.3}s - {winner} won in {moves} moves"
                    )
            else:
                timeouts += 1
            if game:
                file = f"game_{i+1}.pgn"
                Game.save(game, os.path.join(directory, file))
        game_simulator.reset()
    print("----------------------")
    print(
        f"End score\n{e1.name}: {scores[0]}\n{e2.name}: {scores[1]}\nTimeouts: {timeouts}"
    )

# Upload game here to see game evaluation https://lichess.org/paste
