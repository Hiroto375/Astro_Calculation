#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
//  OctreeNode
//
//  ノードは std::vector<OctreeNode> で管理し、
//  子へのアクセスは child[8] に格納したインデックスで行う。
//
//  bit レイアウト (octant 番号):
//    bit2 = x軸  bit1 = y軸  bit0 = z軸   (1 = 上半分)
// ─────────────────────────────────────────────────────────────

struct OctreeNode {
    int    child[8];      // nodes[] へのインデックス; -1 = 子なし
    float  totalMass;
    float3 centerOfMass;
    float3 bmin, bmax;   // このノードが担当する領域
    int    particleCount; // このノードの部分木に含まれる粒子数
    int    particleIdx;   // 葉ノード (粒子1個) のとき >= 0; 内部ノードは -1
};

// ─────────────────────────────────────────────────────────────
//  内部ヘルパー関数
// ─────────────────────────────────────────────────────────────

static OctreeNode makeEmptyNode(float3 bmin, float3 bmax) {
    OctreeNode n;
    for (int i = 0; i < 8; i++) n.child[i] = -1;
    n.totalMass     = 0.0f;
    n.centerOfMass  = {0.0f, 0.0f, 0.0f};
    n.bmin          = bmin;
    n.bmax          = bmax;
    n.particleCount = 0;
    n.particleIdx   = -1;
    return n;
}

// 粒子位置 (px,py,pz) が nodeIdx の中心に対してどの octant に入るかを返す
static int getOctant(int nodeIdx, float px, float py, float pz,
                     const std::vector<OctreeNode>& nodes) {
    float cx = (nodes[nodeIdx].bmin.x + nodes[nodeIdx].bmax.x) * 0.5f;
    float cy = (nodes[nodeIdx].bmin.y + nodes[nodeIdx].bmax.y) * 0.5f;
    float cz = (nodes[nodeIdx].bmin.z + nodes[nodeIdx].bmax.z) * 0.5f;
    return ((px > cx) ? 4 : 0) | ((py > cy) ? 2 : 0) | ((pz > cz) ? 1 : 0);
}

// oct の子ノードが存在しなければ生成する。
// push_back でベクタが再確保されても nodeIdx によるアクセスは有効。
// ただし呼び出し後に nodes[nodeIdx] への既存のポインタ/参照は無効になる可能性がある。
static void ensureChild(int nodeIdx, int oct, std::vector<OctreeNode>& nodes) {
    if (nodes[nodeIdx].child[oct] != -1) return;

    // push_back 前に親の情報を読み出す (参照は push_back で無効になりうる)
    float cx = (nodes[nodeIdx].bmin.x + nodes[nodeIdx].bmax.x) * 0.5f;
    float cy = (nodes[nodeIdx].bmin.y + nodes[nodeIdx].bmax.y) * 0.5f;
    float cz = (nodes[nodeIdx].bmin.z + nodes[nodeIdx].bmax.z) * 0.5f;

    float3 cmin = {
        (oct & 4) ? cx : nodes[nodeIdx].bmin.x,
        (oct & 2) ? cy : nodes[nodeIdx].bmin.y,
        (oct & 1) ? cz : nodes[nodeIdx].bmin.z
    };
    float3 cmax = {
        (oct & 4) ? nodes[nodeIdx].bmax.x : cx,
        (oct & 2) ? nodes[nodeIdx].bmax.y : cy,
        (oct & 1) ? nodes[nodeIdx].bmax.z : cz
    };

    // インデックスを push_back より前に書き込む:
    // push_back で再確保が起きてもデータはコピーされるので値は保持される
    int childIdx = (int)nodes.size();
    nodes[nodeIdx].child[oct] = childIdx;
    nodes.push_back(makeEmptyNode(cmin, cmax));
}

// 粒子 particleIdx を nodeIdx 以下に再帰的に挿入する (ツリー構造のみ構築)
static void insertParticle(int nodeIdx, int particleIdx,
                            std::vector<OctreeNode>& nodes,
                            const std::vector<float4>& particles,
                            int depth) {
    constexpr int MAX_DEPTH = 20; // 完全に一致する座標への無限再帰を防ぐ

    if (nodes[nodeIdx].particleCount == 0) {
        // 空の葉: そのまま粒子を置く
        nodes[nodeIdx].particleIdx   = particleIdx;
        nodes[nodeIdx].particleCount = 1;
        return;
    }

    if (depth >= MAX_DEPTH) {
        // 座標が一致しているとき (現実的には稀) はカウントだけ増やす
        nodes[nodeIdx].particleCount++;
        return;
    }

    if (nodes[nodeIdx].particleIdx >= 0) {
        // 葉ノード (粒子1個) → 既存粒子を子に押し下げて内部ノードに昇格
        int existing = nodes[nodeIdx].particleIdx;
        nodes[nodeIdx].particleIdx = -1; // 内部ノード化

        int oct = getOctant(nodeIdx, particles[existing].x,
                                     particles[existing].y,
                                     particles[existing].z, nodes);
        ensureChild(nodeIdx, oct, nodes);
        insertParticle(nodes[nodeIdx].child[oct], existing, nodes, particles, depth + 1);
    }

    // 新しい粒子を適切な子に挿入
    int oct = getOctant(nodeIdx, particles[particleIdx].x,
                                  particles[particleIdx].y,
                                  particles[particleIdx].z, nodes);
    ensureChild(nodeIdx, oct, nodes);
    insertParticle(nodes[nodeIdx].child[oct], particleIdx, nodes, particles, depth + 1);
    nodes[nodeIdx].particleCount++;
}

// 後順走査で totalMass と centerOfMass を下から積み上げる
static void computeMassCOM(int nodeIdx, std::vector<OctreeNode>& nodes,
                            const std::vector<float4>& particles) {
    if (nodes[nodeIdx].particleIdx >= 0) {
        // 葉ノード: 粒子1個の情報をそのまま使う
        const float4& p = particles[nodes[nodeIdx].particleIdx];
        nodes[nodeIdx].totalMass    = p.w;
        nodes[nodeIdx].centerOfMass = {p.x, p.y, p.z};
        return;
    }

    float  mass = 0.0f;
    float3 com  = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 8; i++) {
        int cidx = nodes[nodeIdx].child[i];
        if (cidx == -1) continue;
        computeMassCOM(cidx, nodes, particles); // 先に子を確定させる
        float cm = nodes[cidx].totalMass;
        com.x += cm * nodes[cidx].centerOfMass.x;
        com.y += cm * nodes[cidx].centerOfMass.y;
        com.z += cm * nodes[cidx].centerOfMass.z;
        mass  += cm;
    }

    nodes[nodeIdx].totalMass = mass;
    if (mass > 0.0f) {
        nodes[nodeIdx].centerOfMass = {com.x / mass, com.y / mass, com.z / mass};
    }
}

// ─────────────────────────────────────────────────────────────
//  公開 API
// ─────────────────────────────────────────────────────────────

std::vector<OctreeNode> buildOctree(const std::vector<float4>& particles) {
    int N = (int)particles.size();
    if (N == 0) return {};

    // 全粒子を包む AABB を計算
    float3 bmin = {particles[0].x, particles[0].y, particles[0].z};
    float3 bmax = bmin;
    for (const auto& p : particles) {
        bmin.x = std::min(bmin.x, p.x);  bmax.x = std::max(bmax.x, p.x);
        bmin.y = std::min(bmin.y, p.y);  bmax.y = std::max(bmax.y, p.y);
        bmin.z = std::min(bmin.z, p.z);  bmax.z = std::max(bmax.z, p.z);
    }

    // 境界上の粒子が丸め誤差で外に出ないよう少し広げる
    const float margin = 1e-4f;
    bmin.x -= margin;  bmin.y -= margin;  bmin.z -= margin;
    bmax.x += margin;  bmax.y += margin;  bmax.z += margin;

    // N 粒子の octree は内部ノード <= N-1 個、葉 = N 個 → 合計 <= 2N-1
    // 余裕を持って 8N 確保して push_back による再確保を最小化する
    std::vector<OctreeNode> nodes;
    nodes.reserve(8 * N);
    nodes.push_back(makeEmptyNode(bmin, bmax));

    for (int i = 0; i < N; i++) {
        insertParticle(0, i, nodes, particles, 0);
    }

    computeMassCOM(0, nodes, particles);
    return nodes;
}

// ─────────────────────────────────────────────────────────────
//  デバッグ用ツリー表示
// ─────────────────────────────────────────────────────────────

static void printTree(int nodeIdx, const std::vector<OctreeNode>& nodes, int depth = 0) {
    const OctreeNode& n = nodes[nodeIdx];
    for (int i = 0; i < depth; i++) std::cout << "  ";
    std::cout << "[" << nodeIdx << "]"
              << " count=" << n.particleCount
              << " mass="  << n.totalMass
              << " com=("  << n.centerOfMass.x << ","
                           << n.centerOfMass.y << ","
                           << n.centerOfMass.z << ")";
    if (n.particleIdx >= 0) std::cout << " leaf(p=" << n.particleIdx << ")";
    std::cout << "\n";
    for (int i = 0; i < 8; i++) {
        if (n.child[i] != -1) printTree(n.child[i], nodes, depth + 1);
    }
}

// ─────────────────────────────────────────────────────────────
//  テスト
// ─────────────────────────────────────────────────────────────

// テスト1: 8粒子、1粒子ずつ各 octant に配置
// → ルートの直下に葉が8枚並ぶ最も単純なケース
static void test_octants() {
    std::vector<float4> p = {
        {-0.5f, -0.5f, -0.5f, 1.0f},  // oct 0
        {-0.5f, -0.5f,  0.5f, 1.0f},  // oct 1
        {-0.5f,  0.5f, -0.5f, 1.0f},  // oct 2
        {-0.5f,  0.5f,  0.5f, 1.0f},  // oct 3
        { 0.5f, -0.5f, -0.5f, 1.0f},  // oct 4
        { 0.5f, -0.5f,  0.5f, 1.0f},  // oct 5
        { 0.5f,  0.5f, -0.5f, 1.0f},  // oct 6
        { 0.5f,  0.5f,  0.5f, 1.0f},  // oct 7
    };

    auto nodes = buildOctree(p);

    // ルートの検証
    assert(nodes[0].particleCount == 8);
    assert(std::abs(nodes[0].totalMass - 8.0f) < 1e-5f);
    // 等質量・点対称 → 重心は原点
    assert(std::abs(nodes[0].centerOfMass.x) < 1e-5f);
    assert(std::abs(nodes[0].centerOfMass.y) < 1e-5f);
    assert(std::abs(nodes[0].centerOfMass.z) < 1e-5f);

    // ルートの子は8個全て存在し、それぞれ葉
    for (int i = 0; i < 8; i++) {
        assert(nodes[0].child[i] != -1);
        int ci = nodes[0].child[i];
        assert(nodes[ci].particleCount == 1);
        assert(nodes[ci].particleIdx   >= 0);
    }

    std::cout << "test_octants PASSED  (nodes=" << nodes.size() << ")\n";
    printTree(0, nodes);
    std::cout << "\n";
}

// テスト2: 異なる質量の2粒子 → 質量加重重心の正確さを確認
static void test_weighted_com() {
    std::vector<float4> p = {
        {1.0f, 0.0f, 0.0f, 1.0f},  // 質量1, x=1
        {3.0f, 0.0f, 0.0f, 3.0f},  // 質量3, x=3
    };

    auto nodes = buildOctree(p);

    assert(nodes[0].particleCount == 2);
    assert(std::abs(nodes[0].totalMass - 4.0f) < 1e-5f);

    // 重心 x = (1*1 + 3*3) / (1+3) = 10/4 = 2.5
    assert(std::abs(nodes[0].centerOfMass.x - 2.5f) < 1e-5f);
    assert(std::abs(nodes[0].centerOfMass.y)         < 1e-5f);
    assert(std::abs(nodes[0].centerOfMass.z)         < 1e-5f);

    std::cout << "test_weighted_com PASSED  (nodes=" << nodes.size() << ")\n";
    printTree(0, nodes);
    std::cout << "\n";
}

// テスト3: 同じ octant に複数粒子が落ちるケース → 深い分割が起きることを確認
static void test_deep_split() {
    // 4粒子が全て正の象限に集まる (ルートの同一 octant に入る)
    std::vector<float4> p = {
        {0.1f, 0.1f, 0.1f, 1.0f},
        {0.1f, 0.1f, 0.9f, 1.0f},
        {0.1f, 0.9f, 0.1f, 1.0f},
        {0.1f, 0.9f, 0.9f, 1.0f},
    };

    auto nodes = buildOctree(p);

    assert(nodes[0].particleCount == 4);
    assert(std::abs(nodes[0].totalMass - 4.0f) < 1e-5f);

    // 等質量 → 重心 = 単純平均
    float ex = (0.1f * 4) / 4.0f;
    float ey = (0.1f + 0.1f + 0.9f + 0.9f) / 4.0f;
    float ez = (0.1f + 0.9f + 0.1f + 0.9f) / 4.0f;
    assert(std::abs(nodes[0].centerOfMass.x - ex) < 1e-5f);
    assert(std::abs(nodes[0].centerOfMass.y - ey) < 1e-5f);
    assert(std::abs(nodes[0].centerOfMass.z - ez) < 1e-5f);

    std::cout << "test_deep_split PASSED  (nodes=" << nodes.size() << ")\n";
    printTree(0, nodes);
    std::cout << "\n";
}

// テスト4: nbody_sim.cu と同じ初期条件 1024粒子
// → 総カウント・総質量の一致を確認
static void test_large() {
    const int N = 1024;
    std::vector<float4> p(N);
    for (int i = 0; i < N; i++) {
        float theta = 2.0f * 3.1415926535f * i / N;
        float r = 1.0f + 0.1f * (i % 10);
        p[i].x = r * std::cos(theta);
        p[i].y = r * std::sin(theta);
        p[i].z = 0.1f * std::sin(3.0f * theta);
        p[i].w = 1.0f;
    }

    auto nodes = buildOctree(p);

    assert(nodes[0].particleCount == N);
    assert(std::abs(nodes[0].totalMass - (float)N) < 1e-2f);

    std::cout << "test_large PASSED  (N=" << N
              << ", nodes=" << nodes.size() << ")\n\n";
}

// ─────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Barnes-Hut Octree Build Tests ===\n\n";
    test_octants();
    test_weighted_com();
    test_deep_split();
    test_large();
    std::cout << "All tests passed.\n";
    return 0;
}
