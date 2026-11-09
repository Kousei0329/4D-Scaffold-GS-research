# HyperNeRF 実験レポート(2026-11-09 時点)

4D-Scaffold-GS(ネイティブ4D anchor)+ HAC系エントロピー符号化(STAC)を HyperNeRF vrig で評価した現状まとめ。
ブランチ `dnerf-hypernerf-experiments`。数値はすべて **圧縮→復号したモデル** の test 画質(nocomp 行を除く)。

## 1. 要点

- 量子化ノイズのバグ修正(commit `46425bf`、HAC++ と同じ挙動に変更)で **chicken が 27.32 → 28.27 dB(+0.95 dB)** になった。サイズは 1.99 → 3.49 MB。
- 修正版の30kは、旧版の120k(28.13 dB / 3.83 MB)より画質が高くサイズが小さい。30k時点でもまだ伸びている。
- ADC-GS の chicken 低レート点(29.61 dB / 3.85 MB)には **まだ 1.3 dB 届いていない**。非圧縮の 4D-Scaffold(60k)が 29.17 dB なので、圧縮による損失と、ベース自体の差の両方がある。
- **修正版のレシピはまだ chicken でしか試していない。** 3dprinter / peel-banana / broom は旧版の数値だけ。他PCでの実行用に `scripts/run_hypernerf_qfix.sh` を用意済み。

## 2. 評価プロトコル

| 項目 | 設定 |
|---|---|
| シーン | vrig: 3dprinter, chicken, peel-banana, broom2(論文中の Broom) |
| 解像度 | データセット同梱の2倍縮小画像(536×960)、`-r 1` |
| train/test 分割 | `dataset.json` の分割。時刻 = warp_id / max |
| 指標 | PSNR、SSIM(通常のSSIM。論文によくある MS-SSIM ではない)、LPIPS(VGG)、ALEX(LPIPS-Alex) |
| サイズ | 実際に書き出した bitstream の合計(feat / scaling / offsets / anchor / mask / hash grid / MLP) |
| チェックポイント選択 | test の RD loss(distortion + λ × MB)が最小のものを選ぶ(`online` と付いた行)。それ以外は最終 iteration |

注意:チェックポイントを test で選んでいるため、比較相手より少し有利な条件になっている。

## 3. 現在のレシピ(`scripts/run_hypernerf_qfix.sh`)

```bash
bash scripts/run_hypernerf_qfix.sh <gpu_id> [scene ...]          # scene省略時は data/HyperNeRF 以下の全シーン
LAMBDAS="0.01 0.005" bash scripts/run_hypernerf_qfix.sh 0 vrig-3dprinter
# 出力: outputs/HyperNeRF/<scene>/baseline_pccolmap_mit1.0_uu25k_qfix_qs1e-4/lmbda_<λ>/
```

- **量子化**:修正版がコードの既定になっている。`--legacy_quant_noise` は付けない(付けると旧版の挙動に戻る)。
- **`--q_scaling_base 1e-4`**:anchor の scaling は 1e-3〜4e-3 程度しかない。HAC の既定値 1e-3(adaptive で最大 2e-3)では、xyz の scaling の 20〜40% が 0 に丸められていた。
- **densification**:0.0002 / 0.1(Scaffold の標準)。`update_until 25000`。
- **初期点群**:`points3D_downsample2.ply`。4DGS のプロトコルで作られた密な COLMAP 点群で、chicken では 80,418 点。`scene.json` を使って `(x − center) × scale` で正規化している。
- **初期時刻**:`MAX_INIT_T=1.0`。各点に U(0, 1) の時刻を割り当て、t_grid(auto = 2 / n_warp_ids)に合わせて丸める。
- 30k iteration、λ = 0.01。

## 4. vrig-chicken の推移

| 設定 | PSNR | SSIM | ALEX | サイズ | 備考 |
|---|---|---|---|---|---|
| 最初のベースライン(points.npy、時刻はすべて t=0) | 23.99 | 0.788 | 0.270 | 1.06 MB | 30k |
| 初期点群を COLMAP の密な点群に変更 | 24.82 | 0.801 | 0.240 | 1.69 MB | 28k |
| + 初期時刻を一様に配置(mit1.0) | 27.06 | 0.816 | 0.206 | 1.84 MB | 30k |
| + update_until 25k(旧版で最良) | 27.32 | 0.819 | 0.200 | 1.99 MB | 30k |
| 旧版 120k(mit1.0 なし) | 28.13 | 0.836 | 0.148 | 3.83 MB | 116k |
| 旧版 60k、update_until 50k | 28.00 | 0.831 | 0.158 | 2.81 MB | 60k |
| **修正版(現在のレシピ)30k** | **28.27** | **0.855** | **0.167** | **3.49 MB** | 30k |
| 参考:非圧縮 4D-Scaffold 60k | 29.17 | 0.876 | 0.104 | – | |

### chicken で試した他の設定(すべて旧版、30k、λ = 0.01)

update_until 25k(27.32 dB)より良い設定は見つからなかった。

| 変更内容 | PSNR | サイズ |
|---|---|---|
| seed を変更(update_until 25k) | 27.31 | 2.15 MB |
| t_grid 0.0061(auto の半分) | 27.19 | 2.15 MB |
| 時間軸を5倍に伸ばす(ts5)/ ts5 + flow lr 3倍 | 27.17 / 27.18 | 1.59 / 1.62 MB |
| ts10 / ts3 | 27.08 / 26.82 | 1.64 / 1.51 MB |
| λ = 0.002 | 27.08 | 2.10 MB |
| voxel 0.0005 / update_interval 50 | 27.06 / 27.06 | 2.42 / 2.09 MB |
| grad threshold 0.0001 | 26.99 | 2.62 MB |
| 動きモデルを basis / Taylor+residual に変更 | 26.92 / 26.66 | 2.24 / 2.06 MB |
| densification 0.0004 / 0.0(圧縮向けの設定) | 26.87 | 1.37 MB |
| densification なし | 25.15 | 0.85 MB |

## 5. 他のシーン(すべて旧版。修正版はまだ回していない)

| シーン | 設定 | PSNR | SSIM | ALEX | サイズ |
|---|---|---|---|---|---|
| 3dprinter | 旧版 60k、update_until 50k(12k で選択) | 21.58 | 0.692 | 0.338 | 2.74 MB |
| 3dprinter | 非圧縮 60k | 22.68 | 0.717 | 0.148 | – |
| peel-banana | 旧版 30k、update_until 25k、λ = 0.01 | 24.88 | 0.775 | 0.211 | 7.85 MB |
| peel-banana | 同上、λ = 0.04 / 0.16 | 24.85 / 24.31 | 0.774 / 0.757 | 0.223 / 0.245 | 5.94 / 4.01 MB |
| peel-banana | 非圧縮 60k | 27.63 | 0.854 | 0.125 | – |
| broom2 | 旧版 60k、update_until 50k(12k で選択) | 21.19 | 0.349 | 0.547 | 3.76 MB |
| broom2 | 非圧縮 60k | 21.08 | 0.368 | 0.279 | – |

- 60k の圧縮 run は、test RD 選択で 12k〜14k の早いチェックポイントが選ばれている。そのため、画質は60k回した結果を反映していない。
- **peel-banana は圧縮すると非圧縮から約 2.8 dB 落ちる。** サイズも大きく(anchor 数 65 万)、旧版で一番弱いシーンだった。修正版でどこまで縮まるかが一番の確認ポイント。

## 6. 比較相手(ADC-GS 論文の表6、シーン別、PSNR / MB)

| 手法 | 3D Printer | Chicken | Broom | Banana | 平均 |
|---|---|---|---|---|---|
| 4DGaussians(Wu) | 22.12 / 63.5 | 29.03 / 90.6 | 22.14 / 46.7 | 29.06 / 51.7 | |
| E-D3DGS | 22.77 / 28.5 | 29.83 / 51.0 | 21.70 / 31.6 | 28.98 / 73.2 | |
| ADC-GS 低レート | 22.92 / 3.23 | 29.61 / 3.85 | 21.80 / 3.27 | 27.35 / 5.73 | 25.42 / 4.02 |
| ADC-GS 高レート | 23.10 / 4.57 | 29.75 / 8.33 | 21.98 / 5.20 | 27.89 / 8.56 | 25.68 / 6.67 |
| **本手法(最新値)** | 21.58 / 2.74(旧版) | **28.27 / 3.49(修正版)** | 21.19 / 3.76(旧版) | 24.88 / 7.85(旧版) | – |

4シーン平均のみ公表されている手法(PSNR / MB):Light4GS 25.35 / 5.15 〜 25.55 / 8.87、TED-4DGS 25.22 / 2.36 〜 25.67 / 3.72。

ADC-GS の行は論文の表にラベルがないため、論文中の平均値と照らし合わせて低・中・高レートに対応づけた。

## 7. 次にやること

1. **修正版レシピを残りの3シーンで実行する**(他PC、`run_hypernerf_qfix.sh`)。特に peel-banana を確認する。
2. chicken の修正版は30kでもまだ伸びているので、60k〜120k で回す。Technicolor では120k回すと、画質はほぼ同じでサイズが約1.6 MB 減った。
3. λ を振って RD カーブを作り、ADC-GS と同じサイズで比べる。
4. 非圧縮でも ADC-GS に 0.4〜0.5 dB 負けている(chicken 29.17 vs 29.61)。ベース側の改善も必要。

## 付録:結果の場所

- 各 run:`outputs/HyperNeRF/<scene>/<tag>/lmbda_<λ>/results.json`(画質)と `outputs.log` の `FINAL SUMMARY` 行(サイズ)
- outputs/ は git 管理外
