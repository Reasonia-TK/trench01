"""回帰テスト: 初版 (c695eb3) と同じ条件の計算が、基準データとビット単位で一致することを確かめる。

基準データ (tests/data/*.npz) は初版のコードで作った。条件は dc 100 eV・マスクなし・40 バッチで、
表面リークなし (base_noleak) と sigma_s = 1e-14 S (base_leak) の 2 つ。今のコードでは、初版と同じ扱いに
なるように bias="dc", mask_t=0, wall_model="absorb" を指定する。

ビット単位の一致を期待できるのは、基準を作ったのと同じ環境 (OS・CPU・uv.lock のバージョン) だけ。
乱数 (PCG64) は環境によらないが、numpy の exp などは CPU の命令によって最後の桁が変わることがあり、
その差が粒子の軌道で広がる。

結果が変わる変更を意図して入れたときだけ、変わった理由を確かめたうえで基準を作り直す:
    uv run python tests/test_regression.py --regen
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # --regen で直接実行しても本体を import できるように
import trench_charging_2d as T  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
CASES = {"base_noleak": 0.0, "base_leak": 1e-14}             # 基準データの名前: sigma_s [S]


def baseline_params(sigma_s):
    return replace(T.Params(), mask_t=0, n_batches=40, bias="dc", sigma_s=sigma_s, wall_model="absorb")


def result_arrays(p):
    """run() の結果を、基準データと同じキーの dict にする。"""
    res = T.run(p, log=lambda s: None)
    return dict(phi=res["phi"], rho=res["rho"], Ex=res["Ex"], Ey=res["Ey"], solid=res["solid"],
                t_conv=res["t_conv"], **{f"hist_{k}": v for k, v in res["hist"].items()},
                **{f"steady_{k}": v for k, v in res["steady"].items()})


@pytest.mark.parametrize("name", CASES)
def test_bitwise_identical_to_baseline(name):
    with np.load(DATA / f"{name}.npz") as d:
        base = {k: d[k] for k in d.files}
    new = result_arrays(baseline_params(CASES[name]))
    assert set(new) == set(base), f"基準データとキーが違います: {sorted(set(new) ^ set(base))}"
    bad = [k for k in base if not np.array_equal(base[k], new[k], equal_nan=True)]
    assert not bad, f"基準データと一致しません: {bad}"


def regenerate(data_dir=DATA):
    """基準データを今のコードで作り直す。"""
    for name, sigma_s in CASES.items():
        path = Path(data_dir) / f"{name}.npz"
        np.savez(path, **result_arrays(baseline_params(sigma_s)))
        print(f"作り直しました: {path}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--regen"]:
        sys.exit("使い方: uv run python tests/test_regression.py --regen  (基準データを今のコードで作り直す)")
    regenerate()
