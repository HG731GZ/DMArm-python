"""以两端逐包 CSV 的会话和实际序号核对双向收发。"""

import argparse
from collections import defaultdict
import csv
from itertools import groupby
from pathlib import Path


def load(path, master):
    commands, feedback = defaultdict(set), defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            event = row["event"].lower()
            if event == ("tx" if master else "rx"):
                table, session, seq = commands, row["master_session"], row["cmd_seq"]
            elif event == ("rx" if master else "tx"):
                table, session, seq = feedback, row["slave_session"], row["fb_seq"]
            else:
                continue
            if session and seq and int(session) and int(seq):
                table[int(session)].add(int(seq))
    return commands, feedback


def ranges(sequences):
    parts = []
    for _, group in groupby(enumerate(sorted(sequences)), lambda item: item[1] - item[0]):
        values = [seq for _, seq in group]
        parts.append(str(values[0]) if len(values) == 1 else f"{values[0]}–{values[-1]}")
    return ", ".join(parts) or "无"


def compare(label, sent, received):
    print(label)
    for session in sorted(sent.keys() | received.keys()):
        tx, rx = sent[session], received[session]
        missing = tx - rx
        common = tx & rx
        print(f"  实发 {len(tx)}  实收去重 {len(rx)}  配对 {len(common)}"
              f"  实发未见接收 {len(missing)} [{ranges(missing)}]"
              f"  仅接收日志出现 {len(rx - tx)} [{ranges(rx - tx)}]")
        if common:
            first, last = min(common), max(common)
            window = {seq for seq in tx if first <= seq <= last}
            print(f"    首末配对序号 [{first}, {last}] 内实发 {len(window)}，"
                  f"缺失 {len(window - rx)} [{ranges(window - rx)}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("master", type=Path, help="主端逐包 CSV")
    parser.add_argument("slave", type=Path, help="从端逐包 CSV，字段约定见 UDP 网络测时说明")
    args = parser.parse_args()
    master_tx, master_rx = load(args.master, True)
    slave_rx, slave_tx = load(args.slave, False)
    compare("主 → 从", master_tx, slave_rx)
    compare("从 → 主", slave_tx, master_rx)
