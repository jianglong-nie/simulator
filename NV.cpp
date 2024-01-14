#include "flow.h"
#include "gpu.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    // gpu的初始化
    int gpuNum = 8;
    int gpuDataSize = 100;
    vector<GPU> gpus;
    for (int i = 0; i < gpuNum; i++) {
        GPU gpu(i, gpuDataSize, false);
        gpus.push_back(gpu);
    }

    // 生成全连接网络拓扑NVLink, 二维数组，NVLink[i][j]表示gpui与相连gpuj的链路带宽
    vector<vector<int>> NVLink(8, vector<int>(8, 10));

    // 实际上8个GPU之间成环连接，gpu0与gpu1相连，gpu1与gpu2相连...，gpu7与gpu0相连
    vector<int> ring = {1, 2, 3, 4, 5, 6, 7, 0};

    // 根据NVLink构建gpus里每个gpu的flows
    for(auto& gpu : gpus) {

        // flow设置，只带NVLink
        int src = gpu.rank;
        int dst = ring[src];
        int dataSize = gpuDataSize;
        string protocol = "NVLink";
        Flow flow;
        flow.init(src, dst, dataSize, protocol);

        int rate = NVLink[src][dst];
        flow.setRate(rate);

        vector<int> path = {src, dst};
        flow.setPath(path);

        // 将flow加入到gpu的flows中
        gpu.flows.push_back(flow);
    }

    // 实现一个discrete-time flow-level的模拟器
    int unitTime = 1;
    int time = 0;
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

    cout << "time: " << time << endl;
    return 0;
}

