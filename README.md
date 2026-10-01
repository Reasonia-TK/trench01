# trench01 — トレンチ構造の表面帯電シミュレーター (2D)

プラズマから飛来する正イオン(高エネルギー・異方的)と電子(低エネルギー・等方的)が、
誘電体トレンチの壁面に電荷を溜めていく様子 (electron shading による帯電) を計算する
2D シミュレーターです。表面に沿った電荷のリーク(表面伝導)も扱えます。

![GUI](docs/gui_screenshot.png)

## ファイル

| ファイル | 内容 |
|---|---|
| `trench_charging_2d.py` | シミュレーション本体 + コマンドライン実行 |
| `trench_charging_gui.py` | Tkinter + matplotlib の GUI (本体を import して使う) |
| `pyproject.toml`, `uv.lock` | 依存パッケージの定義とバージョン固定 (uv) |

## 必要環境

依存パッケージは [uv](https://docs.astral.sh/uv/) で管理しています。

```bash
uv sync          # .venv を作り、uv.lock のバージョンどおりに numpy / scipy / matplotlib を入れる
```

`uv run` で実行すれば `uv sync` は自動で行われます。Python は 3.9 以上で、`.python-version` で 3.12 を
指定しています (なければ uv が自動で入れます)。GUI には Tkinter が必要です (uv が入れる Python には同梱。
Linux のシステム Python では `sudo apt install python3-tk`)。
パッケージの追加は `uv add <名前>`、更新は `uv lock --upgrade` → `uv sync`。
動作確認: Python 3.12 / numpy 2.5 / scipy 1.18 / matplotlib 3.11

## 使い方

```bash
uv run trench_charging_gui.py                       # GUI
uv run trench_charging_2d.py                        # CLI (既定: 30 ms 分の帯電)
uv run trench_charging_2d.py --until_steady         # 飽和帯電まで継続
uv run trench_charging_2d.py --sigma_s 1e-14        # 表面リークあり (シート伝導度 [S])
uv run trench_charging_2d.py --trench_d 160         # 深さ 160 セル = 800 nm (AR=8)
uv run trench_charging_2d.py --help                 # 全パラメータ
```

CLI の出力は `<out>_fields.png`, `<out>_history.png`, `<out>.npz` です。

## モデル

- **構造**: 接地 Si 基板の上の SiO2 (eps_r = 3.9) に中央トレンチ。横は周期境界、上端は φ=0 (または Neumann)。
  既定は 幅 100 nm × 深さ 400 nm (AR=4)、セル 5 nm。
- **粒子**: Monte Carlo テスト粒子。イオンは Ar⁺ 100 eV でほぼ垂直入射、電子は Te=3 eV のマクスウェル分布
  (フラックス重み付き)。両者のフラックスは同じ。固体に当たった粒子は完全吸収。
- **帯電**: 電場を固定して 1 バッチ分の粒子を追跡 → 壁面セルに電荷を蓄積 → Poisson を解き直す、の繰り返し
  (粒子の通過時間 ≪ 帯電の時定数、という準静的近似)。Poisson は誘電率の調和平均を使った有限差分で、
  行列は固定なので LU 分解を 1 回だけ行う。
- **表面リーク**: 表面セルを抵抗網でつなぎ (シート伝導度 `sigma_s`)、電位差に応じて表面に沿って電荷を移動。
  後退オイラーで解くので `sigma_s` が大きくても安定で、総電荷は保存される。
- **飽和判定** (`--until_steady`): 一定バッチごとのブロック平均電位を前のブロックと比較し、
  全観測点で変化が `max(rtol × 最大電位, 3 × 標準誤差)` 以内の状態が連続したら飽和とみなす。

## 結果の例 (AR=4, 飽和まで継続)

| 表面シート伝導度 | 飽和に要した時間 | 底の電位 | 側壁中腹の電位 |
|---|---|---|---|
| なし | 42 ms | 100.4 V | 52.0 V |
| 1e-15 S | 42 ms | 67.4 V | 25.8 V |
| 1e-14 S | 21 ms | 16.2 V | 6.1 V |

リークなしでは底の電位がイオン入射エネルギー (100 eV) で頭打ちになります。

![steady](docs/steady_compare.png)

リークなし・飽和時の電位と電場 (左: 電位、中: 電場、右: 左側壁の表面電位と表面電荷):

![fields](docs/fields_no_leak.png)

## 単純化している点

- イオンの反射、電子の二次放出、シース内の挙動は入れていません。
- リークは表面に沿った伝導のみです。誘電体を貫く膜厚方向のリーク (トンネル電流など) は未実装です。
- 2D (奥行き方向は一様) です。
- 図のラベルは、日本語フォントがない環境でも文字化けしないよう英語にしています。

## 注意

- 1 マクロ粒子が運ぶ電荷が大きいため、個々の表面セルの電位にはショットノイズが乗ります
  (空間的に滑らかな電位分布や飽和値には影響しにくい)。粒子数 (`--n_per_batch`) を増やすと小さくなります。
- 時間刻み `--dt_batch` を大きくしすぎると電位が振動します。
