// __device__ でGPU上で実行することを命令
// float3は3次元ベクトルの構造体
// biは粒子iの位置+質量
// bjは粒子jの位置+加速度
// aiは粒子iの現在の加速度

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai){
    // rは粒子iと粒子jの相対位置
    float3 r;

    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

}