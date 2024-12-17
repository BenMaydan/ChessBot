import chess
from chess.engine import PlayResult, Limit
from lib.engine_wrapper import MinimalEngine
from lib.types import MOVE, HOMEMADE_ARGS_TYPE
import logging
import numpy as np
import os
from lib import model
from lib.config import Configuration
from lib.types import (ReadableType, ChessDBMoveType, LichessEGTBMoveType, OPTIONS_GO_EGTB_TYPE, OPTIONS_TYPE,
                       COMMANDS_TYPE, MOVE, InfoStrDict, InfoDictKeys, InfoDictValue, GO_COMMANDS_TYPE, EGTPATH_TYPE,
                       ENGINE_INPUT_ARGS_TYPE, ENGINE_INPUT_KWARGS_TYPE)
from typing import Any, Optional, Union, Literal, Type, cast


# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)


def material_balance(board: chess.Board) -> int:
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    white_val = (
        chess.popcount(white & board.pawns) +
        3 * chess.popcount(white & board.knights)  +
        3.25 * chess.popcount(white & board.bishops)  +
        5 * chess.popcount(white & board.rooks) +
        9 * chess.popcount(white & board.queens)
    )
    black_val = (
        chess.popcount(black & board.pawns) +
        3 * chess.popcount(black & board.knights) +
        3.25 * chess.popcount(black & board.bishops) +
        5 * chess.popcount(black & board.rooks) +
        9 * chess.popcount(black & board.queens)
    )
    return MaydanEngine.maximizer * (white_val - black_val)


def activity_score(board: chess.Board) -> float:    
    val = 0
    for row in range(0, 8):
      for col in range(0, 8):
        squareIndex = row * 8 + col
        square = chess.SQUARES[squareIndex]
        piece = board.piece_at(square)
        color = board.color_at(square)
        if piece is not None:
            maximizer = MaydanEngine.maximizer if color == chess.WHITE else -MaydanEngine.maximizer
            val += maximizer * MaydanEngine.piece_to_activity_table[piece.piece_type][MaydanEngine.color_indexing[color] * row][col]
    return val / 100.0


def pawns_score(board: chess.Board) -> float:
    val = 0.0
    return val


def heuristic(board: chess.Board) -> float:
    val = 0.0
    val += material_balance(board)
    val += activity_score(board)
    # val += pawns_score(board)
    return val



def max_value(node: chess.Board, depth: int, alpha: float, beta: float) -> float:
    if depth == 0 or node.is_game_over():
        if node.is_checkmate():
            return float("-inf")
        return heuristic(node)
    rv = float("-inf")
    for move in sorted_moves(node):
        new_depth = quiescence_search(node, move, depth)
        node.push(move)
        MaydanEngine.num_evaluated_nodes += 1
        cv = min_value(node, new_depth, alpha, beta)
        node.pop()
        rv = max(rv, cv)
        if rv >= beta:
            return rv
        alpha = max(alpha, rv)
    return rv


def min_value(node: chess.Board, depth: int, alpha: float, beta: float) -> float:
    if depth == 0 or node.is_game_over():
        if node.is_checkmate():
            return float("inf")
        return heuristic(node)
    rv = float("inf")
    for move in sorted_moves(node):
        new_depth = quiescence_search(node, move, depth)
        node.push(move)
        MaydanEngine.num_evaluated_nodes += 1
        cv = max_value(node, new_depth, alpha, beta)
        node.pop()
        rv = min(rv, cv)
        if rv <= alpha:
            return rv
        beta = min(beta, rv)
    return rv


def quiescence_search(node: chess.Board, move: chess.Move, depth: int) -> int:
    # if node.is_capture(move):
    #     to_square = move.to_square
    #     white_protectors = len(node.attackers(chess.WHITE, to_square))
    #     black_protectors = len(node.attackers(chess.BLACK, to_square))
    #     # the idea is to only increase the depth if the capture is "unprotected"
    #     # meaning you don't have enough pieces to capture back after the entire sequence
    #     if white_protectors != black_protectors:
    #         return depth
    #     return depth - 1
    # if node.gives_check(move):
    #     return depth
    return depth - 1


def sorted_moves(board: chess.Board):
    legal_moves = list(board.legal_moves)

    def move_priority(move):
        if board.gives_check(move):
            return 0  # Highest priority: Check
        elif board.is_capture(move):
            return 1  # Second priority: Capture
        elif board.is_attacked_by(board.turn, move.to_square):
            return 2  # Third priority: Threat
        else:
            return 3  # Lowest priority: Other moves

    legal_moves.sort(key=move_priority)
    return legal_moves


class MaydanEngine(MinimalEngine):

    king_table = None
    queen_table = None
    rook_table = None
    bishop_table = None
    knight_table = None
    pawn_table = None

    piece_to_activity_table = {}
    maximizer = 1
    maximizer_mapping = {chess.WHITE: 1, chess.BLACK: -1}
    color_indexing = {chess.WHITE: 1, chess.BLACK: -1}

    num_evaluated_nodes = 0

    def __init__(self, commands: COMMANDS_TYPE, options: OPTIONS_GO_EGTB_TYPE, stderr: Optional[int],
                 draw_or_resign: Configuration, game: Optional[model.Game] = None, name: Optional[str] = None,
                 **popen_args: str):
        super().__init__(commands, options, stderr, draw_or_resign, game, name=name, **popen_args)
        eng_path = os.path.abspath("engines")
        table_path = os.path.join(eng_path, "activity_tables")
        assert os.path.isdir(table_path)
        MaydanEngine.king_table = np.load(os.path.join(table_path, "king_activity_table.npy"))
        MaydanEngine.queen_table = np.load(os.path.join(table_path, "queen_activity_table.npy"))
        MaydanEngine.rook_table = np.load(os.path.join(table_path, "rook_activity_table.npy"))
        MaydanEngine.bishop_table = np.load(os.path.join(table_path, "bishop_activity_table.npy"))
        MaydanEngine.knight_table = np.load(os.path.join(table_path, "knight_activity_table.npy"))
        MaydanEngine.pawn_table = np.load(os.path.join(table_path, "pawn_activity_table.npy"))


        MaydanEngine.piece_to_activity_table[chess.PAWN] = MaydanEngine.pawn_table
        MaydanEngine.piece_to_activity_table[chess.KNIGHT] = MaydanEngine.knight_table
        MaydanEngine.piece_to_activity_table[chess.BISHOP] = MaydanEngine.bishop_table
        MaydanEngine.piece_to_activity_table[chess.ROOK] = MaydanEngine.rook_table
        MaydanEngine.piece_to_activity_table[chess.QUEEN] = MaydanEngine.queen_table
        MaydanEngine.piece_to_activity_table[chess.KING] = MaydanEngine.king_table



    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        return PlayResult(self.find_best_move(board, 4, board.turn), None)
    

    def find_best_move(self, board: chess.Board, depth: int, maximizer: chess.Color) -> MOVE:


        if maximizer == chess.BLACK:
            MaydanEngine.maximizer = -1
        MaydanEngine.num_evaluated_nodes = 0


        rv = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        best_move = None
        for move in sorted_moves(board):


            new_depth = quiescence_search(board, move, depth)
            

            board.push(move)
            cv = min_value(board, new_depth, alpha, beta)
            board.pop()


            if cv >= rv:
                rv = cv
                best_move = move
        

        logger.info("Evaluated {} nodes".format(MaydanEngine.num_evaluated_nodes))
        logger.info("The move with the highest value ({}) is {}".format(rv, best_move))
        

        return best_move
