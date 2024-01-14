import numpy as np
import chess


def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(-1, 8, 8)


def fen_to_bitboard(fen: str, merge_colors=False):
    board = chess.Board(fen)
    black, white = board.occupied_co

    bitboards = np.array(
        [
            black & board.pawns,
            black & board.knights,
            black & board.bishops,
            black & board.rooks,
            black & board.queens,
            black & board.kings,
            # color swap
            white & board.pawns,
            white & board.knights,
            white & board.bishops,
            white & board.rooks,
            white & board.queens,
            white & board.kings,
            # board.castling_rights,
        ],
        dtype=np.uint64,
    )
    array = bitboards_to_array(bitboards)
    if not merge_colors:
        return array
    res = np.zeros((6, 8, 8), dtype=np.int8)
    for i in range(6):
        res[i] += array[i] - array[i + 6]
    return res


def debug_print_boards(fen: str):
    for i, x in enumerate(fen_to_bitboard(fen, True)):
        if i > 0 and i % 8 == 0:
            print()
        if i % 64 == 0:
            print(f"Board {i // 64 + 1}:")
        print(x, end="")

