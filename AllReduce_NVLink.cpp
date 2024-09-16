#include "flow.h"
#include "gpu.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    /*
    参数设置：
    创建，初始化两个服务器集群
    serverGroupNum：每个集群里server数量，选8的倍数比较合适
    gpuNum：每个server里gpu数量
    gpudatasize：每个gpu的待发送数据大小，所有gpu一开始的数据大小都是一样的
    NVLink：每个gpu之间的NVLink带宽 200 GB/s = 204.8 MB/ms = 1638.4 Mb/ms = 1.6384 Mb/(0.001ms = 1μs 1微秒)
    topoBW：每个gpu之间的网络带宽 400 Gb/S = 409.6 Mb/ms = 0.4096 Mb/(0.001ms = 1μs = 1微秒)

    参数换算关系： 
    16MB = 128Mb,
    32MB = 256Mb,
    64MB = 512Mb,
    128MB = 1024Mb,
    256MB = 2048Mb,
    512MB = 4096Mb, 
    1G = 1024MB = 8192Mb, 
    2G = 2048MB = 16384Mb,

    NVLink: 200 GB/s = 204.8 MB/ms = 1638.4 Mb/ms = 1.6384 Mb/(0.001ms = 1μs 1微秒)
    Net: 400 Gb/S = 409.6 Mb/ms = 0.4096 Mb/(0.001ms = 1μs = 1微秒)

    unitTime = 0.001ms = 1μs = 1微秒
    */
    int k = 8;
    int gpuNum = 8;
    float gpuDataSize = 2048;
    float NVLinkBandwidth =  0.98304;
    vector<GPU> gpus;
    for (int i = 0; i < gpuNum; i++) {
        GPU gpu(i, gpuDataSize, false);
        gpus.push_back(gpu);
    }

    // 生成全连接网络拓扑NVLink, 二维数组，NVLink[i][j]表示gpui与相连gpuj的链路带宽
    vector<vector<float>> NVLink(8, vector<float>(8, NVLinkBandwidth));

    // 实际上8个GPU之间成环连接，gpu0与gpu1相连，gpu1与gpu2相连...，gpu7与gpu0相连
    vector<int> ring = {1, 2, 3, 4, 5, 6, 7, 0};

    // 根据NVLink构建gpus里每个gpu的flows
    for(auto& gpu : gpus) {

        // flow设置，只带NVLink
        pair<int, int> src = {0, gpu.rank};
        pair<int, int> dst = {0, ring[src.second]};
        float dataSize = gpuDataSize;
        gpu.dataSize -= dataSize;
        string protocol = "NVLink";
        Flow flow;
        flow.init(src, dst, dataSize, protocol);

        float rate = NVLink[src.second][dst.second];
        flow.setRate(rate);

        vector<int> path = {src.second, dst.second};
        flow.setPath(path);

        // 将flow加入到gpu的flows中
        gpu.flows.push_back(flow);
    }

    // 实现一个discrete-time flow-level的模拟器
    float unitTime = 1;
    float time = 0;
    while (true) {
        // 时间步进
        time += unitTime;

        // 每个gpu执行step
        for (auto& gpu : gpus) {
            gpu.step(unitTime);
        }

        // 每个gpu执行control
        for (auto& gpu : gpus) {
            gpu.control(unitTime);
        }

        // 检查是否所有gpu都完成了计算和通信，结束了工作
        bool isAllFinished = true;
        for (auto& gpu : gpus) {
            if (!gpu.isFinished) {
                isAllFinished = false;
                break;
            }
        }
        if (isAllFinished) {
            break;
        }
    }

    cout << "------------------------------------------" << endl;
    cout << "Total time: " << 2 * (k - 1) * time << endl;
    cout << "Simulation finished!" << endl;
    cout << "------------------------------------------" << endl;
    return 0;
}

