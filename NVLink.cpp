#include "flow.h"
#include "gpu.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    // gpu的初始化
    int gpuNum = 8;
    float gpuDataSize = 200; // 200MB, 500MB, 1G=1024MB, 2G=2048MB
    float NVLinkBandwidth = 4; // 双向NVLink 400GB/s 可以换算为 409.6 MB/ms，近似400MB/ms，40MB/0.1ms
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
            gpu.control();
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
    cout << "time: " << time << endl;
    cout << "Simulation finished!" << endl;
    cout << "------------------------------------------" << endl;
    return 0;
}

