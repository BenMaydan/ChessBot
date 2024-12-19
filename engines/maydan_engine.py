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
    # val += activity_score(board)
    # val += pawns_score(board)
    return val



def max_value(node: chess.Board, depth: int, time_in_qsearch: int, alpha: float, beta: float) -> float:
    capture_moves = [move for move in node.legal_moves if node.is_capture(move)]
    if depth < 0 or (depth == 0 and len(capture_moves) == 0) or node.is_game_over():
        if node.is_checkmate():
            return float("-inf")
        return heuristic(node)
    if depth == 0 and len(capture_moves) > 0:
        time_in_qsearch += 1
    rv = float("-inf")
    for move in sorted_moves(node, depth):
        MaydanEngine.num_evaluated_nodes += 1
        node.push(move)
        new_depth = q_search(node, depth, time_in_qsearch)
        cv = min_value(node, new_depth, time_in_qsearch, alpha, beta)
        node.pop()
        rv = max(rv, cv)
        if rv >= beta:
            return rv
        alpha = max(alpha, rv)
    return rv


def min_value(node: chess.Board, depth: int, time_in_qsearch: int, alpha: float, beta: float) -> float:
    capture_moves = [move for move in node.legal_moves if node.is_capture(move)]
    if depth < 0 or (depth == 0 and len(capture_moves) == 0) or node.is_game_over():
        if node.is_checkmate():
            return float("inf")
        return heuristic(node)
    if depth == 0 and len(capture_moves) > 0:
        time_in_qsearch += 1
    rv = float("inf")
    for move in sorted_moves(node, depth):
        MaydanEngine.num_evaluated_nodes += 1
        node.push(move)
        new_depth = q_search(node, depth, time_in_qsearch)
        cv = max_value(node, new_depth, time_in_qsearch, alpha, beta)
        node.pop()
        rv = min(rv, cv)
        if rv <= alpha:
            return rv
        beta = min(beta, rv)
    return rv


def q_search(node: chess.Board, depth: int, time_in_qsearch: int) -> int:
    # capture_moves = [move for move in node.legal_moves if node.is_capture(move)]
    # if depth == 0:
    #     if time_in_qsearch > MaydanEngine.max_time_in_qsearch or len(capture_moves) == 0:
    #         return -1
    #     return 0
    return depth - 1


def sort_capture_moves(board: chess.Board, capture_moves: list[chess.Move]) -> list[chess.Move]:
    if len(capture_moves) == 0:
        return capture_moves
    
    def move_priority(move):
        capturing_piece: chess.Piece = board.piece_at(move.from_square)
        capturing_value = MaydanEngine.piece_value[capturing_piece.piece_type]

        if board.is_en_passant(move):
            captured_value = MaydanEngine.piece_value[chess.PAWN]
        else:
            captured_piece: chess.Piece = board.piece_at(move.to_square)
            captured_value = MaydanEngine.piece_value[captured_piece.piece_type]

        # lower number is when captured value is higher than capturing value (this is a GOOD capture)
        # think capturing is pawn and captured is queen
        # this results in 1 - 9 = -8 (the BEST kind of capture)
        return capturing_value - captured_value

    capture_moves.sort(key=move_priority)
    return capture_moves


def sorted_moves(board: chess.Board, depth: int) -> list[chess.Move]:
    legal_moves = list(board.legal_moves)
    if depth == 0:
        return sort_capture_moves(board, [move for move in legal_moves if board.is_capture(move)])
    
    pure_check_moves = [move for move in legal_moves if board.gives_check(move) and not board.is_capture(move)]
    capture_moves = [move for move in legal_moves if board.is_capture(move)]
    capture_moves = sort_capture_moves(board, capture_moves)
    rest_moves = [move for move in legal_moves if not board.gives_check(move) and not board.is_capture(move)]

    
    return pure_check_moves + capture_moves + rest_moves


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

    piece_value = {chess.KING: 0, chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.25, chess.ROOK: 5, chess.QUEEN: 9}

    num_evaluated_nodes = 0

    max_time_in_qsearch = 3

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
        for move in sorted_moves(board, depth):
            

            board.push(move)
            new_depth = q_search(board, depth, 0)
            cv = min_value(board, new_depth, 0, alpha, beta)
            board.pop()


            if cv >= rv:
                rv = cv
                best_move = move
        

        logger.info("Evaluated {} nodes".format(MaydanEngine.num_evaluated_nodes))
        logger.info("The move with the highest value ({}) is {}".format(rv, best_move))
        

        return best_move
