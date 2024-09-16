#ifndef NETWORK_H
#define NETWORK_H

#include <map>
#include <vector>
#include <queue>
#include <algorithm>
#include <limits>
#include "flow.h"
#include "server.h"

class Network {
public:
    // 网络中包含两组server集群，每组集群中包含有serverNum个server，每个server中包含8个gpu
    int serverGroupNum;
    int gpuNum;
    float gpuDataSize;
    std::vector<std::vector<float>> NVLink;
    std::vector<Server> serverGroup;

    // 邻接矩阵表示的网络拓扑，只包含Spine和Leaf，不包含Server
    // pair表示两节点之间的链路，first表示link上流的数量flow num（初始化0），second表示link的带宽bandwidth
    int leafNum = 8;
    int spineNum = 8;
    int nodeNum = 0;
    float topoBW = 10;
    std::vector<std::vector<std::pair<int, float>>> topo; 

    /*
    netflow管理器：记录网络中所有的flow，指针指向network里所有的flow，包括gpu里的flow和背景流量
    */
    std::vector<Flow*> gpuFlowManager;
    std::vector<Flow*> bgFlowManager;

    // Network构造函数
    Network();

    // Network的初始化函数，生成网络拓扑，有8个leaf，8个spine，彼此是全互连的。
    void init(int serverGroupNum, int gpuNum, float gpuDataSize, std::vector<std::vector<float>> NVLink, float topoBW);

    void ECMPRandom();

    // 用来测试的路由函数，针对8server，每个server里8GPU，8Leaf，8Spine的网络拓扑
    void RoutingRandom(int gpuFlowRoutingNum);

    void Routing();

    // water-filling算法用来求出leafs里每个flow的rate
    void waterFilling();
    void waterFilling2();

    // dijkstra算法用来求出src与dst之间的最短路径
    std::vector<int> dijkstra(int srcId, int dstId);

    // step()函数，用来模拟网络中的每个节点的计算和通信
    void step(float unitTime);

    // control()函数，用来模拟网络中的每个节点的控制器
    void control(float unitTime);
};

#endif // NETWORK_H