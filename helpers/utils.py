import json
import time


def parse_position(line):
    d = json.loads(line)
    best_eval = max(d["evals"], key=lambda x: x["depth"])
    first_pvs = best_eval["pvs"][0]
    eval = first_pvs["cp"] if "cp" in first_pvs else first_pvs["mate"]
    is_mate = 1 if "mate" in first_pvs else 0
    return f'{eval}, {d["fen"].strip()}, {first_pvs["line"].split()[0]}, {best_eval["depth"]}, {is_mate}'


def parse_dataset(games_to_load: int, save_path: str, starts: str):
    t0 = time.time()
    with open("data/lichess_db_eval.json", "r") as f:
        with open(save_path, "w") as fw:
            i = 0
            fw.write("eval,fen,next_move,depth,is_mate\n")
            while i < games_to_load:
                line = f.readline()
                try:
                    x = parse_position(line)
                    if x.split(", ")[1].split()[1] == starts:
                        fw.write(x + "\n")
                        i += 1
                except Exception as e:
                    print(f"Exception {e} while parsing line {line}")
                if i * 10 % games_to_load == 0:
                    print(f"{i * 100 / games_to_load}%")
    print(f"Done in {time.time() - t0:.2f}s")
