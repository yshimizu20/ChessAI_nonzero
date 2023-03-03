import chess.pgn
import sys
import re

if __name__ == "__main__":
    date_str = sys.argv[1]
    # check if data_str is yyyy-mm format using regex
    if not re.match(r"\d{4}-\d{2}", date_str):
        raise ValueError("Invalid date format. Please use yyyy-mm format.")

    i = j = 0
    with open(f"lichess_db_standard_rated_{date_str}.pgn", 'r') as rf:
        with open(f"processed/output_{date_str}.pgn", 'w') as wf:
            while True:
                game = chess.pgn.read_game(rf)
                if game is None:
                    break
                # Calculate the average ELO of the two players
                try:
                    white_elo = int(game.headers["WhiteElo"])
                    black_elo = int(game.headers["BlackElo"])
                except ValueError:
                    continue
                avg_elo = (white_elo + black_elo) // 2

                j += 1

                # Filter out games with an average ELO below 2000
                if avg_elo < 2000:
                    continue
            
                i += 1
                if i % 100 == 0:
                    print(i, j)

                wf.write(str(game))