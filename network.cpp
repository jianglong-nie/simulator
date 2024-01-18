#include "network.h"

/*
// struct FlowInfo的init函数的实现
void FlowInfo::init(std::pair<int, int> src, std::pair<int, int> dst, int dataSize) {
    this->src = src;
    this->dst = dst;
    this->dataSize = dataSize;

}
*/

// Network构造函数的实现
Network::Network() {}

// init函数的实现
void Network::init(int serverGroupNum, int gpuNum, float gpuDataSize, std::vector<std::vector<float>> NVLink, float topoBW) {
    this->serverGroupNum = serverGroupNum;
    this->gpuNum = gpuNum;
    this->gpuDataSize = gpuDataSize;
    this->NVLink = NVLink;
    this->topoBW = topoBW;
    int nodeNum = leafNum + spineNum;

    // 初始化topoSpineLeaf为nodeNum x nodeNum的零矩阵
    topoSpineLeaf = std::vector<std::vector<std::pair<int, float>>>(nodeNum, std::vector<std::pair<int, float>>(nodeNum, {0, 0}));

    // 连接每个leaf节点到所有的spine节点
    for (int leaf = 0; leaf < leafNum; ++leaf) {
        for (int spine = leafNum; spine < nodeNum; ++spine) {
            topoSpineLeaf[leaf][spine] = {0, topoBW}; // 初始化流数量为0，带宽为topoBW
            topoSpineLeaf[spine][leaf] = {0, topoBW}; // 初始化流数量为0，带宽为topoBW
        }
    }

    // 初始化serverGroup1和serverGroup2
    for (int i = 0; i < serverGroupNum; i++) {
        serverGroup1.push_back(Server(i, gpuNum, gpuDataSize, NVLink));
    }
    for (int i = serverGroupNum; i < serverGroupNum * 2; i++) {
        serverGroup2.push_back(Server(i, gpuNum, gpuDataSize, NVLink));
    }

    /*
    初始化leaf
    leaf[0]的dst是leaf[1], leaf[1]的dst是leaf[2]，以此类推，直到leaf[7]的dst是leaf[0]
    leaf[8]的dst是leaf[9], leaf[9]的dst是leaf[10]，以此类推，直到leaf[15]的dst是leaf[8]

    得到每个leaf从src_id到dst_id的所有可能的path, 用paths记录
    16个leaf[0-15]，8个spine[16-23]，leaf跟spine彼此是全互连的，leaf之间没有连接，spine之间没有连接
    */

    for (int i = 0; i < leafNum; i++) {
        Leaf leaf;
        leaf.src_id = i;
        if (i == 7) {
            leaf.dst_id = 0;
        } else if (i == 15) {
            leaf.dst_id = 8;
        } else {
            leaf.dst_id = i + 1;
        }
        this->leafs.push_back(leaf);
    }

    for (auto& leaf : leafs) {
        for (int i = 16; i < nodeNum; i++) {
            std::vector<int> path;
            path.push_back(leaf.src_id);
            path.push_back(i);
            path.push_back(leaf.dst_id);
            leaf.paths.push_back(path);
        }
    }
}

/*
ECMP函数的实现
目标：
    把每个leaf里的所有flow分配到合适的路径上，基于flow.dataSize和path的带宽来分配
思路：
    1. 遍历所有的flow, 得到dataSum为所有flow的dataSize之和
    2. 基于leaf的paths和topoSpineLeaf，计算出每个path的带宽，存储到pathsBandwidth中，以及求出带宽之和pathsBWSum
    3. 设计一个变量pathsDataSize，用来记录该路径上的最大可用dataSize，初始化为pathsDataSize = dataSum * pathsBandwidth / pathsBWSum
    4. 遍历所有的flow，flow先分配到paths[0]，如果paths[0]的pathsDataSize不够，就分配到paths[1]，以此类推
*/
void Network::ECMP() {
    
    for (auto& leaf : leafs) {
        // 1. 遍历所有的flow, 得到dataSum为所有flow的dataSize之和
        float dataSum = 0;
        for (auto& flow : leaf.flows) {
            dataSum += flow.dataSize;
        }

        // 2. 基于leaf的paths和topoSpineLeaf，计算出每个path的带宽，存储到pathsBandwidth中，以及求出带宽之和pathsBWSum
        std::vector<float> pathsBandwidth;
        float pathsBWSum = 0;

        for (auto& path : leaf.paths) {
            float pathBW = FLOAT_MAX;
            for (int i = 0; i < path.size() - 1; i++) {
                float curPathBW = topoSpineLeaf[path[i]][path[i + 1]].second;
                pathBW = std::min(pathBW, curPathBW);
            }
            pathsBandwidth.push_back(pathBW);
            pathsBWSum += pathBW;
        }

        // 3. 设计一个数组pathsDataSize，用来记录每个路径上的最大可用dataSize，初始化为pathsDataSize[i] = dataSum * pathsBandwidth[i] / pathsBWSum
        std::vector<float> pathsDataSize;
        for (auto& pathBW : pathsBandwidth) {
            float dataSize = (dataSum * pathBW + pathsBWSum - 1) / pathsBWSum;
            pathsDataSize.push_back(dataSize);
        }

        /* 
        4. 遍历所有的flows，
            (1) 为flow分配合适路径：flow[0]先分配到paths[0]，然后分配flows[1], flows[2]到paths[0]上
            直到分配flows[i]时，paths[0]上的pathsDataSize降为0或者为负，flow[i]还是会分配到paths[0]，
            但是下一个flow[i + 1]将分配到paths[1]，以此类推
            (2) 给gpu的flow也分配合适的路径
            (3) 更新topoSpineLeaf里的流数量
        */
        int index = 0;
        for (auto& flow : leaf.flows) {
            // 如果pathsDataSize[index]为0或者为负，就将index加1
            while (index < pathsDataSize.size() && pathsDataSize[index] <= 1) {
                index++;
            }
            // 如果index超过了pathsDataSize的大小，就将所有未分配的flow都分配到pathsDataSize的最后一位
            if (index >= pathsDataSize.size()) {
                index = pathsDataSize.size() - 1;
            }
            // 1. 为flow分配合适路径
            // 将flow分配到paths[index]上
            // 将flow的src里表示的server id gpu rank里flow的path也更新一下
            std::vector<int> path = leaf.paths[index];
            flow.setPath(path);
            
            // 2. 给gpu的flow也分配合适的路径
            // 将flow的src里表示的 {serverId, gpuRank} 里flows[1]的path也更新一下
            int serverId = flow.src.first;
            int gpuRank = flow.src.second;
            
            // 如果server id < serverGroupNum, 说明是serverGroup1里的server。反之，则是serverGroup2里
            if (serverId < serverGroupNum) {
                auto& gpu = serverGroup1[serverId].gpus[gpuRank];
                gpu.flows[1].setPath(path);
            } else {
                auto& gpu = serverGroup2[serverId - serverGroupNum].gpus[gpuRank];
                gpu.flows[1].setPath(path);
            }

            // 3. 更新topoSpineLeaf里的流数量
            for (int i = 0; i < path.size() - 1; i++) {
                topoSpineLeaf[path[i]][path[i + 1]].first++;
            }

            // 更新pathsDataSize[index]
            pathsDataSize[index] -= flow.dataSize;
        }
    }
}

/*
目标：
    实现ECMPRandom函数，将每个leaf里的所有flow，以及对应gpu里的flow随机分配到一个路径上
思路：
    1. 遍历每个leaf所有的flow，针对该flow的src，确定它属于哪个gpu
    2. 针对该gpu，随机选择一个leaf.paths，将flow分配到该path上
    3. 更新topoSpineLeaf里的流数量
*/

void Network::ECMPRandom() {
    for (auto& leaf : leafs) {
        for (auto& flow : leaf.flows) {
            // 1. 遍历每个leaf所有的flow，针对该flow的src，确定它属于哪个gpu
            int serverId = flow.src.first;
            int gpuRank = flow.src.second;
    
            GPU* gpu = nullptr;
            if (serverId < serverGroupNum) {
                gpu = &serverGroup1[serverId].gpus[gpuRank];
            } else {
                gpu = &serverGroup2[serverId - serverGroupNum].gpus[gpuRank];
            }

            // 2. 针对该gpu，随机选择一个leaf.paths，leaf.paths总共有8个，将flow分配随机分配到这8个path上
            // 保证index的值在0-7之间
            int index = rand() % leaf.paths.size();
            std::vector<int> path = leaf.paths[index];
            gpu->flows[1].setPath(path);


            // 3. 更新topoSpineLeaf里的流数量
            for (int i = 0; i < path.size() - 1; i++) {
                topoSpineLeaf[path[i]][path[i + 1]].first++;
            }
        }
    }
}


/*
目标：
    实现waterFilling函数，求出每个flow的rate
思路：
    1. 遍历两个serverGroup里gpu的flows[1]，求出每个flows[1]的rate（因为flows[1]代表Net的流）
    2. rate根据flow的path和topoSpineLeaf里的topoBW/流数量来计算
*/

void Network::waterFilling() {
    // 1. 遍历两个serverGroup里gpu的所有的flow，求出每个flow的rate
    for (auto& server : serverGroup1) {
        for (auto& gpu : server.gpus) {
            Flow& flow = gpu.flows[1];
            // 2. rate根据flow的path和topoSpineLeaf里的topoBW/流数量来计算，如果liu num为0，就设置rate为topoBW
            float rate = FLOAT_MAX;
            for (int i = 0; i < flow.path.size() - 1; i++) {
                int flowNum = topoSpineLeaf[flow.path[i]][flow.path[i + 1]].first;
                float linkBW = topoSpineLeaf[flow.path[i]][flow.path[i + 1]].second;
                rate = std::min(rate, flowNum == 0 ? linkBW : linkBW / flowNum);
            }
            flow.setRate(rate);
        }
    }

    for (auto& server : serverGroup2) {
        for (auto& gpu : server.gpus) {
            Flow& flow = gpu.flows[1];
            // 2. rate根据flow的path和topoSpineLeaf里的topoBW/流数量来计算，如果liu num为0，就设置rate为topoBW
            float rate = FLOAT_MAX;
            for (int i = 0; i < flow.path.size() - 1; i++) {
                int flowNum = topoSpineLeaf[flow.path[i]][flow.path[i + 1]].first;
                float linkBW = topoSpineLeaf[flow.path[i]][flow.path[i + 1]].second;
                rate = std::min(rate, flowNum == 0 ? linkBW : linkBW / flowNum);
            }
            flow.setRate(rate);
        }
    }
}

// setp函数的实现，执行每个server里的step函数
void Network::step(float unitTime) {
    for (auto& server : serverGroup1) {
        server.step(unitTime);
    }
    for (auto& server : serverGroup2) {
        server.step(unitTime);
    }
}

// control函数的实现，执行每个server里的control函数
void Network::control() {
    for (auto& server : serverGroup1) {
        server.control();
    }
    for (auto& server : serverGroup2) {
        server.control();
    }
}

    /*
    目标：
    计算每个server外部的Net的flow的path和rate

    连接关系：
    对于serverGroup1里64个服务器，每个server里的8个gpu编号从0-7，每个server的gpu0都与leaf0相连，gpu1都与leaf1相连，以此类推
    对于serverGroup2里64个服务器，每个server里的8个gpu编号从0-7，每个server的gpu0都与leaf8相连，gpu1都与leaf9相连，以此类推
    
    因此对于flowNet的path而言，serverGroup1里服务器连到topoSpineLeaf的0-7，serverGroup2里服务器连到topoSpineLeaf的8-15
    而topoSpineLeaf中leaf(0-7)与spine(16-23)全互连，leaf(8-15)与spine(16-23)全互连

    因此对于serverGroup1里的服务器，flowNet的path为src -> leaf -> spine -> leaf -> dst

    leafs[0]记录了serverGroup1里所有服务器的gpu0的flow, leafs[7]记录了serverGroup1里所有服务器的gpu7的flow
    leafs[8]记录了serverGroup2里所有服务器的gpu0的flow, leafs[15]记录了serverGroup2里所有服务器的gpu7的flow
    
    首先得确定该flow的path，再根据Net中的water-filling算法求出rate

    */