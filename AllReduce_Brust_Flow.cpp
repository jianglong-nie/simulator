#include "flow.h"
#include "gpu.h"
#include "server.h"
#include "network.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <string>

using namespace std;

/*
提前确定好背景流量的数量，每个背景流量的srcId和dstId及path
brust flow: 周期性的增加flow.dataSize，每个周期增加 (0 - gpuDataSize * 1/50)的随机值
5个flow:
flow0: srcId = 26, dstId = 0, path = {26, 66, 72, 64, 0};
flow1: srcId = 10, dstId = 33, path = {10, 66, 73, 65, 33};
flow2: srcId = 5, dstId = 14, path = {5, 69, 74, 70, 14};
flow3: srcId = 27, dstId = 20, path = {27, 67, 75, 68, 20};
flow4: srcId = 46, dstId = 79, path = {46, 70, 79};
*/

/*
生成背景流量的两个函数
1. 自己定义，专门针对8server 8gpu
2. 随机生成3个flow
*/

void generateBrustFlow(std::vector<Flow>& bgFlows, Network& network) {
    std::vector<int> leafIdList = {64, 65, 66, 67, 68, 69, 70, 71};
    std::vector<int> spineIdList = {72, 73, 74, 75, 76, 77, 78, 79};

    Flow flow1(64, 66, 0, "Net");
    Flow flow2(67, 68, 0, "Net");
    Flow flow3(69, 71, 0, "Net");
    Flow flow4(68, 70, 0, "Net");
    Flow flow5(65, 67, 0, "Net");

    flow1.setPath({64, 73, 66});
    flow2.setPath({67, 75, 68});
    flow3.setPath({69, 77, 71});
    flow4.setPath({68, 72, 70});
    flow5.setPath({65, 74, 67});

    bgFlows.push_back(flow1);
    bgFlows.push_back(flow2);
    bgFlows.push_back(flow3);
    bgFlows.push_back(flow4);
    bgFlows.push_back(flow5);

    for (auto& flow : bgFlows) {
        network.bgFlowManager.push_back(&flow);
    }
}

void generateBrustFlowRandom(std::vector<Flow>& bgFlows, Network& network, int bgFlowNum, int bgFlowRoutingNum) {
    std::vector<int> leafIdList = {64, 65, 66, 67, 68, 69, 70, 71};
    std::vector<int> spineIdList = {72, 73, 74, 75, 76, 77, 78, 79};

    for (int i = 0; i < bgFlowNum; i++) {
        int srcLeafId = leafIdList[rand() % leafIdList.size()];
        int dstLeafId = leafIdList[rand() % leafIdList.size()];
        while (srcLeafId == dstLeafId) {
            dstLeafId = leafIdList[rand() % leafIdList.size()];
        }
        Flow flow(srcLeafId, dstLeafId, 0, "Net");

        int spineId;
        int count = 0;
        do {
            if(count > 100) {
                break;
            }
            spineId = spineIdList[rand() % spineIdList.size()];
            count++;
        } while (network.topo[srcLeafId][spineId].first >= bgFlowRoutingNum || network.topo[spineId][dstLeafId].first >= bgFlowRoutingNum);

        std::vector<int> path = {srcLeafId, spineIdList[rand() % spineIdList.size()], dstLeafId};
        flow.setPath(path);
        bgFlows.push_back(flow);
    }

    for (auto& flow : bgFlows) {
        network.bgFlowManager.push_back(&flow);
    }
}

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
    /*
    All2All
    7stage：
    1: 7
    2: 4 （2*3 + 1） 
    4: 2  (4 + 3)
    8: 1  (7*1)
    */
    int k = 8; // k = gpuNum
    int stageNum = 2 * (k - 1); // stageNum = 14
    float unitTime = 1; // unitTime = 0.001ms = 1μs = 1微秒
    float time = 0;
    float maxTime = 0;

    // 每条路由路径上，gpuFlow的数量最多为gpuFlowRoutingNum
    int gpuFlowRoutingNum = 1;

    // 背景流量的参数设置
    // 每条路由路径上，（背景流+gpu流）的数量最多为bgFlowRoutingNum
    // n * unitTime
    // {0 ~ gpuDataSize * 10 / bgFlowDataSizeRatio }
    int bgFlowNum = 10;
    int bgFlowRoutingNum = 2; 
    float bgFlowPeriod =  300;
    int bgFlowDataSizeRatio = 550;

    for (int stage = 1; stage <= 1; stage++) {

        int serverGroupNum = 8;
        int gpuNum = 8;
        float gpuDataSize = 2048;
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
        float ratio = 0.83;
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

        // 创建，初始化背景流量。背景流量的srcId和dstId及path都是提前确定好的
        // 但是每隔一段周期，背景流量的dataSize会随机增加 为0-0.1倍的gpuDataSize
        vector<Flow> bgFlows;
        generateBrustFlowRandom(bgFlows, network, bgFlowNum, bgFlowRoutingNum);
        //generateBrustFlow(bgFlows, network);

        network.waterFilling();

        // 实现一个discrete-time flow-level的模拟器
        while (true) {
            // 时间步进
            time += unitTime;

            // 每个gpu执行step
            network.step(unitTime);

            // 周期性插入一个brust flow
            if (fmod(time, bgFlowPeriod) == 0) {
                for (auto& flow : bgFlows) {
                    flow.dataSize += gpuDataSize * (rand() % 10) / bgFlowDataSizeRatio;
                }
            }

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
        
        for (auto& server : network.serverGroup) {
            for (auto& gpu : server.gpus) {
                // 针对sever里的gpu打印出flows里的completionTime和sentDataSize，flows[0]是nvlink，flows[1]是net
                cout << "serverId: " << server.id << " gpuId: " << gpu.rank << endl;
                for (auto& flow : gpu.flows) {
                    cout << flow.protocol << ": ";
                    cout << "completionTime = " << flow.completionTime << " sentdataSize = " << flow.sentDataSize << endl;
                    if (flow.completionTime > maxTime) {
                            maxTime = flow.completionTime;
                    }
                }
            }
        }
        cout << "------------------------------------------" << endl;
        cout << "One stage time: " << time << endl;
        cout << "maxTime: " << maxTime << endl;
        cout << "------------------------------------------" << endl;
        
    }
    cout << "------------------------------------------" << endl;
    cout << "Total time: " << time*14 << endl;
    cout << "Simulation finished!" << endl;
    cout << "------------------------------------------" << endl;
    return 0;
}