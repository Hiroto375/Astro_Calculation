# Astro_Calculation — プロジェクト概要

GPU計算自主ゼミ用のN体重力シミュレーション実装集。
CUDA (`.cu`) ファイル群と可視化スクリプト (`Plot.py`) で構成される。

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `check_cuda.cu` | CUDA デバイス認識確認用の最小コード |
| `N_body.cu` | 共有メモリタイル計算を使った O(N²) 重力カーネル（1ステップのみ） |
| `N_body_nc.cu` | `N_body.cu` にデバッグログを追加したバリアント |
| `nbody_sim.cu` | オイラー積分で時間発展させ `trajectory.csv` を出力するフルシミュ |
| `Plot.py` | `trajectory.csv` → `solar_animation.mp4` アニメーション生成 |
| `barnes_hut.cu` | **Barnes-Hut 法の実装（進行中）** |

---

## Barnes-Hut 実装状況

### 完了: Octree 構築（CPU 側のみ）

`barnes_hut.cu` に以下を実装済み。

**データ構造**
- `OctreeNode` 構造体
  - `child[8]` — 子ノードへのインデックス（`-1` = 子なし）
  - `totalMass`, `centerOfMass` — 部分木の総質量・重心
  - `bmin`, `bmax` — このノードが担う空間領域
  - `particleCount` — 部分木の粒子数
  - `particleIdx` — 葉ノードのみ `>= 0`（内部ノードは `-1`）
- ノードは `std::vector<OctreeNode>` で管理（配列ベース）

**実装済み関数**
- `buildOctree()` — 公開 API。AABB 計算 → 粒子挿入 → 重心計算を一括実行
- `insertParticle()` — 再帰的な粒子挿入。葉の分割・子ノードの遅延生成を行う
- `computeMassCOM()` — 後順走査で `totalMass` / `centerOfMass` を積み上げる
- `ensureChild()` — 指定 octant の子ノードを必要に応じて生成
- `printTree()` — デバッグ用ツリー表示

### 未実装

- `computeForce()` — θ 基準によるツリー走査と近似力計算
- オイラー積分・CSV 出力の組み込み（`nbody_sim.cu` から流用予定）
- GPU カーネル化

---

## テスト結果

`barnes_hut.cu` の `main()` に 4 つのテストを内蔵。CPU のみでコンパイル・実行済み。

| テスト | 内容 | 結果 |
|---|---|---|
| `test_octants` | 8粒子・各 octant に1個 → 根直下に葉8枚 | PASSED (9ノード) |
| `test_weighted_com` | 質量1と3の2粒子 → 重心 x=2.5 の確認 | PASSED (3ノード) |
| `test_deep_split` | 同じ x 座標に4粒子 → y/z で正しく分割 | PASSED (5ノード) |
| `test_large` | 1024粒子（円盤配置） → 総質量・総カウント一致 | PASSED (1663ノード) |

---

## 次にやること

1. **`computeForce()` の実装**
   - ツリーを再帰的に走査し、開口角 θ = s/d で近似判定
   - 葉ノードまたは遠距離セルの重心・質量で加速度を計算

2. **`N_body.cu` との数値照合**
   - θ = 0（全展開）にすれば O(N²) と同じ結果になるはずなので一致を確認

3. **性能比較**
   - θ を上げて O(N log N) に切り替え、`nbody_sim.cu` と実行時間を比較

4. **GPU 移植**（並列化フェーズ）
   - このフェーズで `cuda_runtime.h` を本物に戻す
   - ツリーを配列ベースにしたのは GPU 移植を見越した設計

---

## 注意点

### コンパイル（現在）
ローカル環境に `nvcc` がないため、CPU テストは以下で実行：

```bash
# /tmp/cuda_runtime.h にスタブを作成してから
c++ -x c++ -o barnes_hut_test barnes_hut.cu -std=c++17 -I/tmp
```

スタブ (`/tmp/cuda_runtime.h`) の内容:
```cpp
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
inline float3 make_float3(float x, float y, float z) { return {x,y,z}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x,y,z,w}; }
```

GPU 環境では `-I/tmp` を除いてそのまま `nvcc` でコンパイルできる。

### `ensureChild()` の push_back 問題
`nodes.push_back()` はベクタの再確保を引き起こす可能性があるため、
**参照ではなくインデックスで `nodes[nodeIdx]` にアクセスすること**。
`nodes[nodeIdx].child[oct] = childIdx` は push_back より前に書き込んでいるが、
再確保時にデータはコピーされるため値は保持される。この順序を変えないこと。

### ノード数の上限
ツリーのノード数は理論上 `2N - 1` 以下（各内部ノードが2子以上を持つ場合）。
現在は `nodes.reserve(8 * N)` で確保。`MAX_DEPTH = 20` で完全一致座標への無限再帰を防いでいる。

### `Plot.py` との互換性
`nbody_sim.cu` が出力する CSV の列名 `step,id,x,y,z` に合わせれば `Plot.py` をそのまま流用できる。
