#include "flow.h"
#include "gpu.h"
#include "server.h"
#include "network.h"
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

    200 GB/s = 204.8 MB/ms = 1638.4 Mb/ms = 1.6384 Mb/(0.001ms = 1μs 1微秒)
    400 Gb/S = 409.6 Mb/ms = 0.4096 Mb/(0.001ms = 1μs = 1微秒)

    unitTime = 0.001ms = 1μs = 1微秒
    */

    int serverGroupNum = 8;
    int gpuNum = 8;
    float gpuDataSize = 8064;
    float NVLinkBandwidth = 1.6384;
    float topoBW = 0.4096;
    std::vector<std::vector<float>> NVLink(gpuNum, std::vector<float>(gpuNum, NVLinkBandwidth));

    // 创建，初始化网络
    Network network;
    network.init(serverGroupNum, gpuNum, gpuDataSize, NVLink, topoBW);

    // 

    /*

    给每个server的里的8个gpu创建两个flow
    一个flow是基于Server内部NVLink和ring的flow
    一个flow是基于Server外部Net的flow
    每个flow的dataSize根据server里的ratio比例来分配

    目标：
    1. 设置好每个server内部的NVLink和ring的flow
    2. 初始化每个server外部的Net的flow的基本信息，但是不知道该flow的path和rate
    
    */
    // ratio = NVLink / (NVLink + Net)
    float ratio = 0.8;
    for (auto& server : network.serverGroup) {
        // 对每个server应该调用一下flow distribution函数，计算一下分配给NVLink和Net的数据大小，或者比例
        for (auto& gpu : server.gpus) {
            // flow设置，只带NVLink
            pair<int, int> src = {server.id, gpu.rank};
            pair<int, int> dst = {server.id, server.ring[src.second]};

            // 得到基于NVLink的flow
            Flow flowNVLink;
            float dataSizeNV = gpu.dataSize * ratio;
            gpu.dataSize -= dataSizeNV;
            flowNVLink.init(src, dst, dataSizeNV, "NVLink");

            float rateNV = server.NVLink[src.second][dst.second];
            flowNVLink.setRate(rateNV);
            gpu.flows.push_back(flowNVLink);

            // 得到基于Net的flow, 
            Flow flowNet;
            float dataSizeNet = gpu.dataSize;
            gpu.dataSize -= dataSizeNet;
            flowNet.init(src, dst, dataSizeNet, "Net");
            // 将flow加入到gpu的flows中
            gpu.flows.push_back(flowNet);

            // 对于gpuFlowManager，只需要将gpu里基于Net的flow加入到gpuFlowManager中即可
            network.gpuFlowManager.push_back(&gpu.flows[1]);
        }
    }

    network.Routing();
    network.waterFilling();

    // 实现一个discrete-time flow-level的模拟器
    float unitTime = 1; // unitTime = 0.001ms = 1μs = 1微秒
    float time = 0;
    while (true) {
        // 时间步进
        time += unitTime;

        // 每个gpu执行step
        network.step(unitTime);

        // 每个gpu执行control
        network.control(unitTime);

        // 检查是否所有server里的所有gpu是否都完成了计算和通信，结束了工作
        bool isAllFinished = true;
        for (auto& server : network.serverGroup) {
            for (auto& gpu : server.gpus) {
                if (!gpu.isFinished) {
                    isAllFinished = false;
                    break;
                }
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