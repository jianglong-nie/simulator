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
    NVLink：每个gpu之间的NVLink带宽 400 GB/s = 409.6 MB/ms = 3276.8 Mb/ms = 32.768 Mb/(0.01ms)
    topoBW：每个gpu之间的网络带宽 400 Gb/S = 409.6 Mb/ms = 4.096 Mb/(0.01ms)

    参数换算关系： 
    200MB = 1600Mb, 
    500MB = 4000Mb, 
    1G = 1024MB = 8192Mb, 
    2G = 2048MB = 16384Mb
    400 GB/S = 400 *1024 MB/ 1000ms = 409.6 MB/ms = 409.6 * 8 Mb/ms = 3276.8 Mb/ms = 32.768 Mb/0.01ms
    400 Gb/S = 400 *1024 Mb/ 1000ms = 409.6 Mb/ms = 4.096 Mb/(0.01ms)
    */

    int serverGroupNum = 8;
    int gpuNum = 8;
    float gpuDataSize = 16384;
    float NVLinkBandwidth = 32.768;
    float topoBW = 4.096;
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
    float ratio = 0.88;
    for (auto& server : network.serverGroup1) {
        // 对每个server应该调用一下flow distribution函数，计算一下分配给NVLink和Net的数据大小，或者比例
        for (auto& gpu : server.gpus) {
            // flow设置，只带NVLink
            pair<int, int> src = {server.id, gpu.rank};
            pair<int, int> dst = {server.id, server.ring[src.second]};

            // 得到基于NVLink的flow
            Flow flowNVLink;
            float dataSizeNV = gpu.dataSize * ratio;
            // gpu.dataSize -= dataSizeNV;
            flowNVLink.init(src, dst, dataSizeNV, "NVLink");

            float rateNV = server.NVLink[src.second][dst.second];
            flowNVLink.setRate(rateNV);
            gpu.flows.push_back(flowNVLink);

            // 得到基于Net的flow, 
            Flow flowNet;
            float dataSizeNet = gpu.dataSize * (1 - ratio);
            // gpu.dataSize -= dataSizeNet;

            flowNet.init(src, dst, dataSizeNet, "Net");

            // 对于group1,gpu.rank对应着leafs的编号，都是0-7。将flowNet插入到leafs里
            network.leafs[gpu.rank].flows.push_back(flowNet);


            // 将flow加入到gpu的flows中
            gpu.flows.push_back(flowNet);
        }
    }

    for (auto& server : network.serverGroup2) {
        // 对每个server应该调用一下flow distribution函数，计算一下分配给NVLink和Net的数据大小，或者比例
        // ratio = NVLink / (NVLink + Net) = 0.8
        for (auto& gpu : server.gpus) {
            // flow设置，只带NVLink
            pair<int, int> src = {server.id, gpu.rank};
            pair<int, int> dst = {server.id, server.ring[src.second]};

            // 得到基于NVLink的flow
            Flow flowNVLink;
            float dataSizeNV = gpu.dataSize * ratio;
            // gpu.dataSize -= dataSizeNV;
            flowNVLink.init(src, dst, dataSizeNV, "NVLink");

            float rateNV = server.NVLink[src.second][dst.second];
            flowNVLink.setRate(rateNV);

            // 得到基于Net的flow, 首先得确定该flow的path，再根据Net中的water-filling算法求出rate
            Flow flowNet;
            float dataSizeNet = gpu.dataSize * (1 - ratio);
            // gpu.dataSize -= dataSizeNet;

            flowNet.init(src, dst, dataSizeNet, "Net");

            // 对于group2, gpu.rank对应着leafs的编号 + 8，从8-15。将flowNet插入到leafs里
            network.leafs[gpu.rank + 8].flows.push_back(flowNet);


            // 将flow加入到gpu的flows中
            gpu.flows.push_back(flowNVLink);
            gpu.flows.push_back(flowNet);
        }
    }

    network.ECMPRandom();
    network.waterFilling();

    // 实现一个discrete-time flow-level的模拟器
    float unitTime = 1; // 0.01ms
    float time = 0;
    while (true) {
        // 时间步进
        time += unitTime;

        // 每个gpu执行step
        network.step(unitTime);

        // 每个gpu执行control
        network.control();

        // 检查是否所有gpu都完成了计算和通信，结束了工作
        bool isAllFinishedGroup1 = true;
        bool isAllFinishedGroup2 = true;
        for (auto& server : network.serverGroup1) {
            for (auto& gpu : server.gpus) {
                if (!gpu.isFinished) {
                    isAllFinishedGroup1 = false;
                    break;
                }
            }
        }
        for (auto& server : network.serverGroup2) {
            for (auto& gpu : server.gpus) {
                if (!gpu.isFinished) {
                    isAllFinishedGroup2 = false;
                    break;
                }
            }
        }
        if (isAllFinishedGroup1 && isAllFinishedGroup2) {
            break;
        }
    }
    cout << "------------------------------------------" << endl;
    cout << "time: " << time << endl;
    cout << "Simulation finished!" << endl;
    cout << "------------------------------------------" << endl;
    return 0;
}