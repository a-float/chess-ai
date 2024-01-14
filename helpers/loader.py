import os
import berserk
import chess
from chess import InvalidMoveError, IllegalMoveError, AmbiguousMoveError
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("LICHESS_TOKEN")
LEADERBOARD_PERF_TYPE = "rapid"
GAMES_PERF_TYPE = "rapid,classical"
MAX_GAMES = 2

class Loader:
    def __init__(self, token):
        self.valid = 0
        self.invalid = 0
        session = berserk.TokenSession(token=token)
        self.client = berserk.Client(session=session)

    def game_to_fen(self, game: list) -> list:
        moves = game["moves"].split()
        board = chess.Board()
        fen = []
        
        for move in moves:
            board.push_san(move)
            fen.append(board.fen())
        
        return fen
    
    def parse_analysis(self, game: list) -> list:
        parsed_analyses = []
        for analysis in game["analysis"]:
            if 'eval' in analysis.keys():
                parsed_analyses.append({'eval': analysis['eval']})
            else:
                parsed_analyses.append({'mate': analysis['mate']})
            
        return parsed_analyses
       

    def get_player_games(self, player_id: any) -> list:
        games = self.client.games.export_by_player(player_id, analysed=True, evals=True, perf_type=GAMES_PERF_TYPE, max=MAX_GAMES)
        
        parsed_games = []
        for game in games:
            try:
                analysis = self.parse_analysis(game)
                moves = self.game_to_fen(game)[:len(analysis)]
                parsed_games.append(moves, analysis)

                self.valid += 1

            except (InvalidMoveError, IllegalMoveError, AmbiguousMoveError) as error:
                print(error)
                # print(game["moves"])
                # print("-----------------------------")
                self.invalid += 1
                
        return parsed_games
    
    def get_player_ids(self, players_no: int) -> list:
        players = self.client.users.get_leaderboard(perf_type=LEADERBOARD_PERF_TYPE, count=players_no)
        players_ids = [player["id"] for player in players]
        return players_ids
        
    def get_games(self, player_ids: list) -> list:
        games = []
        for player_id in player_ids:
            games += self.get_player_games(player_id)

        return games

if __name__ == '__main__':
    loader = Loader(API_TOKEN)
    ids = loader.get_player_ids(players_no=2)
    result = loader.get_games(player_ids=ids)
    print(result)

    print(f"valid: {loader.valid}, invalid: {loader.invalid}")
