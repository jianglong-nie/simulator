#ifndef NETWORK_H
#define NETWORK_H

#include <map>
#include <vector>
#include "flow.h"
#include "server.h"

struct Leaf {
    // leaf的id
    int src_id;
    // leaf中所有gpu的flow的目的地
    int dst_id;
    // 记录流过该leaf的所有flow
    std::vector<Flow> flows;
    
    // 记录每个leaf交换机的所有可能的path
    // path[0]记录了所有可能的path，leafPath[0][0]记录了leaf0的第一条path
    std::vector<std::vector<int>> paths;
};


class Network {
public:
    // 网络中包含两组server集群，每组集群中包含有serverNum个server，每个server中包含8个gpu
    int serverGroupNum;
    int gpuNum;
    float gpuDataSize;
    std::vector<std::vector<float>> NVLink;
    std::vector<Server> serverGroup1;
    std::vector<Server> serverGroup2;

    // 邻接矩阵表示的网络拓扑，只包含Spine和Leaf，不包含Server
    // pair表示两节点之间的链路，first表示link上流的数量flow num（初始化0），second表示link的带宽bandwidth
    int leafNum = 16;
    int spineNum = 8;
    float topoBW = 10;
    std::vector<std::vector<std::pair<int, float>>> topoSpineLeaf; 

    // leaf交换机共16个，前8个leaf[0-7]连接serverGroup1中的servers，后8个leaf[8-15]连接serverGroup2中的servers
    // leaf[0]记录了serverGroup1里所有服务器的gpu0的flow, leaf[7]记录了serverGroup1里所有服务器的gpu7的flow
    // leaf[8]记录了serverGroup2里所有服务器的gpu0的flow, leaf[15]记录了serverGroup2里所有服务器的gpu7的flow
    std::vector<Leaf> leafs;

    // Network构造函数
    Network();

    // Network的初始化函数，生成网络拓扑，有16个leaf(0-15)，8个spine（16-23），彼此是全互连的。
    void init(int serverGroupNum, int gpuNum, float gpuDataSize, std::vector<std::vector<float>> NVLink, float topoBW);

    // ECMP算法用来求出leafs里每个flow的路由
    void ECMP();

    void ECMPRandom();

    // water-filling算法用来求出leafs里每个flow的rate
    void waterFilling();

    // step()函数，用来模拟网络中的每个节点的计算和通信
    void step(float unitTime);

    // control()函数，用来模拟网络中的每个节点的控制器
    void control();
};

#endif // NETWORK_H