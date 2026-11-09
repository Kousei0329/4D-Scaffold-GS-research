# HyperNeRF 実験レポート(2026-11-09 時点)

4D-Scaffold-GS(ネイティブ4D anchor)+ HAC系エントロピー符号化(STAC)を HyperNeRF vrig で評価した現状まとめ。
ブランチ `dnerf-hypernerf-experiments`。数値はすべて **圧縮→復号したモデル** の test 画質(nocomp 行を除く)。

## 1. 要点

- 量子化ノイズのバグ修正(commit `46425bf`、HAC++ と同じ挙動に変更)で **chicken が 27.32 → 28.27 dB(+0.95 dB)** になった。サイズは 1.99 → 3.49 MB。
- 修正版の30kは、旧版の120k(28.13 dB / 3.83 MB)より画質が高くサイズが小さい。30k時点でもまだ伸びている。
- ADC-GS の chicken 低レート点(29.61 dB / 3.85 MB)には **まだ 1.3 dB 届いていない**。非圧縮の 4D-Scaffold(60k)が 29.17 dB なので、圧縮による損失と、ベース自体の差の両方がある。
- **修正版のレシピはまだ chicken でしか試していない。** 3dprinter / peel-banana / broom は旧版の数値だけ。他PCでの実行用に `scripts/run_hypernerf_qfix.sh` を用意済み。
- **他PC(RTX 5080)で、修正版のレシピを 4 シーン × 5 つの λ、120k で実行中**(2026-11-09 09:20 UTC 開始)。詳細は 9 章。→ **8 章の調査のため 10:08 UTC に中断**(chicken λ = 0.01 が 19k の時点、途中結果は破棄)。まだ再開していない。
- 同じレシピの 30k を他PCで回したところ、chicken の λ = 0.01 は test RD 選択で **12k が選ばれて 27.42 dB** になった。30k 時点の圧縮前の評価は 28.66 dB だった。チェックポイント選択の影響がかなり大きい(9.3 節)。
- **非圧縮で負けている原因の調査(8 章)**:peel-banana は過学習ではなく表現力不足(train と test の差 0.17 dB、30k でもまだ伸びている)。`--motion_sigma_reg_weight 0.01` で非圧縮 +0.43 dB、圧縮後 +0.59 dB だが、anchor が約 3 割増えてサイズも増える。broom2 / chicken の「時間窓が狭い」仮説は外れ(`TIME_SCALE=0.5` で悪化。chicken の旧版で時間軸を伸ばした ts3 / 5 / 10 も悪化していた)。broom2 は train と test の差が約 2 dB あり、別の原因がありそう。

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

## 8. 圧縮なしで負けている原因の調査

(他PC の RTX 5080、2026-11-09 実行)

非圧縮の 4D-Scaffold が変形ベースの手法(4DGaussians / E-D3DGS / ADC-GS)に負けている理由を、2 つの仮説で調べた。

- **H1(peel-banana)**:anchor や時間方向の表現力が足りない。
- **H2(broom2 / chicken)**:anchor の時間窓が狭すぎて、時間方向にうまく補間できていない。

### 8.1 設定

- 共通:`COMPRESS=0 GRAD_TH=0.0002 PHOTO_TH=0.1 MAX_INIT_T=1.0 ITERATIONS=30000`、EXTRA に必ず `--update_until 25000` を入れた。
- 1 GPU で 1 本ずつ順番に回した。実行スクリプトは `/workspace/tools/run_hypernerf_nocomp_study.sh` で、他PC にだけ置いてある。
- 評価方法は 2 章のまま変えていない(536×960、SSIM、ALEX)。非圧縮の run は 30k のチェックポイントで評価した。
- 量子化まわりの既定値は変えていない。`--legacy_quant_noise` は使っていない。モデルのコードも変えておらず、すべて既存のフラグと環境変数で試した。
- T_GRID の auto 値は 2 / warp id の数。peel-banana 0.00390、broom2 0.01015、chicken 0.01220。

**H2-a の設定についての注意。** 指示では `lambda_temporal_sigma` / `lambda_sigma_multiply` で時間窓を広げることになっていた。しかしコードを確認すると、この 2 つは densification 統計の重み付けにしか使われていなかった。

```
weight = opacity_t * (sigma * lambda_sigma_multiply) ** lambda_temporal_sigma + weight_base   # scene/gaussian_model.py:1693
```

時間窓の幅は `opacity_t = exp(-|Δt·σ|^β)` の σ で決まる。σ は cov MLP が Gaussian ごとに学習している(gaussian_renderer/__init__.py:407-409)。そこで H2-a では代わりに `TIME_SCALE=0.5` を使った。これは `--frame_ratio 0.5` にあたり、時間軸を半分に縮める。同じ σ で 2 倍のフレームをカバーするので、時間窓を実質 2 倍に広げたことになる。ただし flow の 1 フレームあたりの移動量も半分になるので、純粋に時間窓だけを変えた実験ではない。

### 8.2 結果(非圧縮、30k)

| シーン | 設定 | train PSNR | test PSNR | 差 | SSIM | ALEX | anchor 数 | 学習時間 |
|---|---|---|---|---|---|---|---|---|
| peel-banana | 基準(uu25k) | 26.32 | 26.15 | 0.17 | 0.825 | 0.168 | 923,536 | 36 分 |
| peel-banana | H1-a:T_GRID 0.00195(auto の半分) | 26.51 | 26.18 | 0.33 | 0.827 | 0.165 | 1,110,930 | 41 分 |
| peel-banana | H1-b:GRAD_TH 0.0001 | 26.87 | 26.49 | 0.38 | **0.837** | **0.154** | 1,827,749 | 81 分 |
| peel-banana | H1-c:`--motion_sigma_reg_weight 0.01` | **26.98** | **26.58** | 0.40 | 0.836 | 0.157 | 1,217,118 | 49 分 |
| broom2 | 基準(uu25k) | 23.35 | 21.37 | 1.98 | 0.379 | 0.355 | 544,487 | 27 分 |
| broom2 | H2-a:TIME_SCALE 0.5 | 23.54 | 21.16 | 2.38 | 0.374 | **0.317** | 626,839 | 35 分 |
| broom2 | H2-b:`--motion_smooth_weight 0.05` | 23.50 | **21.39** | 2.11 | **0.386** | 0.329 | 624,020 | 32 分 |
| chicken | 基準(uu25k) | 30.04 | 28.32 | 1.72 | 0.858 | 0.151 | 160,509 | 14 分 |
| chicken | H2-a:TIME_SCALE 0.5 | 29.90 | 28.26 | 1.64 | **0.864** | 0.140 | 195,797 | 15 分 |
| chicken | H2-b:`--motion_smooth_weight 0.05` | **30.32** | **28.43** | 1.89 | 0.861 | **0.138** | 192,340 | 16 分 |

- train PSNR は学習中の評価(`[ITER 30000] Evaluating train`)の値。test PSNR / SSIM / ALEX は `results.json` の値。差は train − test。
- anchor 数は 30k の `point_cloud.ply` の点数。
- 落ちた run はなかった。

学習中の test PSNR の推移:

| シーン / 設定 | 7.5k | 15k | 22.5k | 30k |
|---|---|---|---|---|
| peel-banana 基準 | 23.94 | 25.08 | 25.63 | 26.16 |
| peel-banana H1-c | 24.20 | 25.45 | 26.10 | 26.59 |
| broom2 基準 | 21.38 | **21.52** | 21.39 | 21.38 |
| broom2 H2-b | 21.32 | **21.49** | 21.42 | 21.39 |
| chicken 基準 | 25.46 | 27.31 | 27.97 | 28.33 |

broom2 はどの設定でも 15k が最高で、そのあと test が下がった。

### 8.3 圧縮ありでの確認(λ = 0.01、30k)

効果があった設定を、3 章のレシピ(`--update_until 25000 --q_scaling_base 0.0001`)に加えて回した。`run_hypernerf_qfix.sh` は書き換えず、同じ設定を `train_hypernerf.sh` に直接渡した。実行スクリプトは `/workspace/tools/run_hypernerf_comp_check.sh`(他PC のみ)。

| シーン | 設定 | 選ばれた iteration | PSNR(復号後) | SSIM | ALEX | サイズ | anchor 数 | 30k の test PSNR(圧縮前) | 30k の train PSNR | 30k の実ビットストリーム |
|---|---|---|---|---|---|---|---|---|---|---|
| peel-banana | レシピのまま | 12k | 24.34 | 0.772 | 0.224 | 10.15 MB | 572,692 | 25.72 | 25.18 | 12.07 MB |
| peel-banana | + msr 0.01 | 12k | **24.92** | **0.784** | **0.216** | 13.49 MB | 749,148 | **26.33** | 25.89 | 17.11 MB |
| chicken | レシピのまま(9.3 節と同じ run) | 12k | 27.42 | 0.833 | 0.211 | **2.83 MB** | 153,426 | **28.66** | 29.92 | 3.74 MB |
| chicken | + msr 0.01 | 16k | **27.97** | 0.844 | 0.188 | 3.00 MB | 151,566 | 28.49 | 29.68 | 3.61 MB |
| chicken | + smooth 0.05 | 18k | 27.59 | **0.846** | **0.186** | 3.27 MB | 158,060 | 28.19 | 29.42 | 3.64 MB |

- 「選ばれた iteration」は 2 章の方式(test の RD loss で online best)で選んだもの。
- PSNR / SSIM / ALEX / サイズ / anchor 数は、そのチェックポイントを圧縮して復号したモデルの値。
- peel-banana の + msr 0.01 は、どちらの run も同じ 12k が選ばれた。画質は +0.59 dB 上がったが、サイズも 33% 増えた。30k の時点でも、圧縮前で +0.61 dB、ビットストリームで +42%。同じサイズで比べて良くなったとは言えない。
- chicken の + msr 0.01(+0.55 dB)と + smooth 0.05(+0.17 dB)は、**選ばれた iteration が違う**(12k / 16k / 18k)。30k の圧縮前で比べると、どちらもレシピのままより低い(−0.17 / −0.47 dB)。圧縮後の差はチェックポイント選択によるもので、設定そのものの効果とは言えない。
- 非圧縮の run(8.2 節)と比べると、圧縮した run の 30k の圧縮前 PSNR は peel-banana で 0.4 dB 低く(25.72 vs 26.16)、chicken では 0.3 dB 高い(28.66 vs 28.33)。

### 8.4 結論

**H1(peel-banana は表現力不足):おおむね当たり。**
- train と test の差は 0.17 dB しかない。過学習ではなく、train の画質そのものが足りていない。
- 30k の時点でも test は 7.5k ごとに約 0.5 dB 伸びている。30k では学習が足りていない。
- anchor が増える設定(H1-b / H1-c)では、train と test がそろって 0.3〜0.4 dB 上がった。
- 一番良かったのは H1-c(+0.43 dB)。ただし anchor も 32% 増えたので、正則化そのものが効いたのか、点が増えたから上がったのかは切り分けられていない。
- H1-b は anchor が 2 倍、学習時間が 2.3 倍になり、効率が悪い。
- T_GRID を半分にした H1-a はほぼ効果がなかった(+0.03 dB)。
- それでも 26.6 dB 前後で、ADC-GS / 4DGaussians の 27.4〜29 dB とはまだ差がある。

**H2(broom2 / chicken は時間窓が狭い):外れ。**
- 時間窓を広げた H2-a は、2 シーンとも test PSNR が下がった(broom2 −0.22、chicken −0.06)。broom2 では train と test の差がかえって広がった。
  - chicken の旧版で試した ts3 / 5 / 10(時間軸を伸ばす = 時間窓を狭める方向、4 章)も悪化していたので、時間窓の幅は今の値が最適に近いと考えられる。
- H2-b は chicken で +0.11 dB、ALEX 0.151 → 0.138。broom2 では PSNR が同じで、ALEX だけ 0.355 → 0.329 に良くなった。
  - 効果は小さいが、知覚的な指標(ALEX)はどちらのシーンでも改善した。
- broom2 の約 2 dB の差は、train の画質が足りないことが原因ではない。test 視点への汎化の問題で、test は 15k で頭打ちになる。時間窓や動きの滑らかさとは別の原因(視点への過学習、broom の細い穂先の表現など)を疑うべき。

**圧縮ありで取り入れるべき設定:今回の範囲では見つからなかった。**
- msr 0.01 は peel-banana で画質が上がるが、サイズも同じくらい増える。
- chicken の改善は、チェックポイント選択の影響と区別できない。

### 8.5 次に試すこと

1. **peel-banana を 60k〜120k の非圧縮で回す。** 30k の時点でもまだ伸びているので、学習の長さが一番の要因かどうかを確かめる。
2. **msr の効果と anchor 増加を切り分ける。**
   - msr 0.01 と、GRAD_TH を少し上げて anchor 数を基準にそろえたものを比べる。
   - あわせて msr を 0.003 / 0.03 で振る。
3. **broom2 の汎化を調べる。**
   - per_view.json で、どの test 視点・時刻で崩れているかを見る。
   - `update_until` を 15k にする、densification を早めに止める、opacity reset を入れる、などで過学習を抑えて比べる。
4. **圧縮ありの比較で、チェックポイント選択の影響を取り除く。** 30k の固定チェックポイントでも圧縮・復号して評価し、online best と並べる。評価方法自体は変えず、参考値として載せる。

### 8.6 結果の場所(他PC、outputs/ は git 管理外)

- 非圧縮:`outputs/HyperNeRF/<scene>/baseline_pccolmap[_tg0.00195]_mit1.0[_gth0.0001|_ts0.5]_uu25k[_msr0.01|_ms0.05]_nocomp/nocomp/`
- 圧縮あり:`outputs/HyperNeRF/<scene>/baseline_pccolmap_mit1.0_uu25k_qfix_qs1e-4[_msr0.01|_ms0.05]/lmbda_0.01/`
- ログ:`/workspace/data/HyperNeRF/_logs/study/*.log`

## 9. 他PC(RTX 5080)での実行状況(2026-11-09 09:35 UTC 時点)

7 章の 1〜3(残り 3 シーン、120k、λ を振る)をまとめて実行している。

### 9.1 実行内容

```bash
LAMBDAS="0.01 0.05"      ITERATIONS=120000 bash scripts/run_hypernerf_qfix.sh 0 vrig-chicken vrig-3dprinter broom2 vrig-peel-banana
LAMBDAS="0.02 0.03 0.04" ITERATIONS=120000 bash scripts/run_hypernerf_qfix.sh 0 vrig-chicken vrig-3dprinter broom2 vrig-peel-banana
# 出力: outputs/HyperNeRF/<scene>/baseline_pccolmap_it120000_mit1.0_uu25k_qfix_qs1e-4/lmbda_<λ>/
```

- 3 章のレシピをそのまま使い、`ITERATIONS=120000` だけ変えている。
- `update_until` はレシピの EXTRA で 25000 に固定されているので、120k でも 25k のまま。この扱いは未検討。
- 先に λ = 0.01 / 0.05 を 4 シーンで回し、そのあと 0.02 / 0.03 / 0.04 を回す。合計 20 本。
- 1 本約 1.5〜2 時間(chicken は 18〜40 it/s)なので、全部終わるのは 11/10〜11/11 ごろの見込み。
- 2026-11-09 09:35 時点では chicken の λ = 0.01 が 19k / 120k まで進んでいる。120k の結果はまだ出ていない。
- **10:08 UTC に 8 章の調査のため中断した。** 途中の run は `outputs_failed/hypernerf_120k_stopped/` に移した。20 本とも未実行の扱い。

### 9.2 データ準備

- 画像は公式リリース(google/hypernerf v0.1)の `vrig_*.zip` を使った。
- `points3D_downsample2.ply` は、4DGS の README にある配布ファイルをそのまま使った(Google Drive の `hypernerf-pcd.zip`、ID `1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5`)。

| 配布ファイル内のフォルダ | 置き場所 | 点数 |
|---|---|---|
| `virg/virg-chickchicken` | `vrig-chicken` | 80,418 |
| `virg/virg-3dprinter` | `vrig-3dprinter` | 96,466 |
| `virg/broom2` | `broom2` | 38,569 |
| `virg/peel-banana` | `vrig-peel-banana` | 99,108 |

- どのシーンも、`points.npy` と座標の範囲(5〜95 パーセンタイル)が重なることを確認した。
- 最初は COLMAP を回して自分で点群を作っていた(4DGS の手順で 40k 点以下に間引いたもの。chicken は 36,793 点)。配布ファイルとは別物なので使うのをやめた。
  - その点群で学習した run は他PCの `outputs_failed/hypernerf_selfmade_colmap_ply/` に移した。
  - 参考までに、自作の点群での chicken 30k は、λ = 0.01 が 28.03 dB / 2.67 MB、λ = 0.05 が 27.41 dB / 1.41 MB(どちらも 30k が選ばれた)。

### 9.3 他PCでの 30k の結果(120k に切り替える前に回した分)

| シーン | 初期点群 | λ | 選ばれた iteration | PSNR | SSIM | ALEX | サイズ |
|---|---|---|---|---|---|---|---|
| chicken | 配布版(80,418 点) | 0.01 | 12k(online) | 27.42 | 0.833 | 0.211 | 2.83 MB |

学習中に評価した test PSNR(圧縮前)の推移:

| iteration | 7.5k | 15k | 22.5k | 30k |
|---|---|---|---|---|
| PSNR | 26.20 | 27.84 | 28.33 | 28.66 |

- 3 章と同じレシピなのに、4 章の 28.27 dB / 3.49 MB(30k)より 0.85 dB 低い。違いは選ばれたチェックポイントで、こちらは online best に 12k が選ばれた。
- 30k 時点の圧縮前の評価は 28.66 dB なので、最終のモデルを使えば 28.3 dB 前後かそれ以上が期待できる。
- 120k の run では、online best とは別に最終のチェックポイントでも評価する予定。

## 付録:結果の場所

- 各 run:`outputs/HyperNeRF/<scene>/<tag>/lmbda_<λ>/results.json`(画質)と `outputs.log` の `FINAL SUMMARY` 行(サイズ)
- outputs/ は git 管理外
