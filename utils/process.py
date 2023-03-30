import sys
import chess.pgn


def main(year, month):
    i = 0
    j = 0
    k = 0

    with open(f"filtered/output-{year}_{month}.pgn", "r") as rf:
        with open(f"cleaned/output-{year}_{month}.pgn", "w") as wf:
            while True:
                i += 1
                g = chess.pgn.read_game(rf)
                if g is None:
                    break
                try:
                    # if int(g.headers["TimeControl"].split("+")[0]) >= 300:
                    #     j += 1
                    #     wf.write(str(g) + "\n\n")
                    white_elo = int(g.headers["WhiteElo"])
                    black_elo = int(g.headers["BlackElo"])
                    if white_elo >= 2200 and black_elo >= 2200:
                        j += 1
                        wf.write(str(g) + "\n\n")

                except:
                    k += 1

    print("Total games:", i)
    print("Games with time control >= 300:", j)
    print("Games with no time control:", k)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process.py year month")
        exit(1)

    main(sys.argv[1], sys.argv[2])
