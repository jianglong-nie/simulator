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
    int nodeNum = (serverGroupNum * gpuNum) + leafNum + spineNum;

    /*
    目标：
        初始化网络拓扑topo，设置网络带宽
    思路：
        1. 初始化topo为nodeNum x nodeNum的零矩阵
        2. 初始化serverGroup里的gpu与leaf的连接：
            每个server有gpuNum = 8个gpu且gpuNum = leafNum = 8，gpu0与leaf0相连，gpu1与leaf1相连，以此类推
            gpu之间没有连接，只有gpu与leaf之间有连接
        3. leaf与spine之间全连接，但leaf之间没有连接，spine之间没有连接

    */
    // 1. 初始化topo为nodeNum x nodeNum的零矩阵
    topo = std::vector<std::vector<std::pair<int, float>>>(nodeNum, std::vector<std::pair<int, float>>(nodeNum, {0, 0}));
    
    // 2. 初始化serverGroup里的gpu与leaf的连接：
    for (int serverId = 0; serverId < serverGroupNum; ++serverId) {
        for (int gpuRank = 0; gpuRank < gpuNum; ++gpuRank) {
            int gpuId = serverId * gpuNum + gpuRank;
            int leafId = serverGroupNum * gpuNum + gpuRank;

            topo[gpuId][leafId] = {0, topoBW}; // 初始化流数量为0，带宽为NVLink[gpu][gpu]
            topo[leafId][gpuId] = {0, topoBW}; // 初始化流数量为0，带宽为NVLink[gpu][gpu]
        }
    }

    // 3. leaf与spine之间全连接，但leaf之间没有连接，spine之间没有连接
    for (int leaf = 0; leaf < leafNum; ++leaf) {
        for (int spine = 0; spine < spineNum; ++spine) {
            int leafId = serverGroupNum * gpuNum + leaf;
            int spineId = serverGroupNum * gpuNum + leafNum + spine;

            topo[leafId][spineId] = {0, topoBW}; // 初始化流数量为0，带宽为topoBW
            topo[spineId][leafId] = {0, topoBW}; // 初始化流数量为0，带宽为topoBW
        }
    }



    // 初始化serverGroup
    for (int i = 0; i < serverGroupNum; i++) {
        serverGroup.push_back(Server(i, gpuNum, gpuDataSize, NVLink));
    }
}

/*
目标：
    实现ECMPRandom函数，根据gpuFlowManager将gpu里的flow随机分配到一个路径上
思路：
    1. 遍历gpuFlowManager所有的flow，确定flow属于哪个gpu，针对该flow的src，dst换算出在topo上的节点编号
    2. 针对该gpu，随机选择一条从src到dst的可行路径，将flow分配到这条路径上。路径通常有5个节点，起点，三中间节点，终点
    3. 更新topo里的流数量
*/

void Network::ECMPRandom() {
    for(auto& flow : gpuFlowManager) { // 注意这里的flow是指针类型
        // 1. 遍历gpuFlowManager所有的flow，确定flow所属的gpu，针对该flow的src，dst换算出在topo上的节点编号
        int serverIdSrc = flow->src.first;
        int gpuRankSrc = flow->src.second;

        int serverIdDst = flow->dst.first;
        int gpuRankDst = flow->dst.second;

        int srcId = serverIdSrc * gpuNum + gpuRankSrc;
        flow->srcId = srcId;
        int dstId = serverIdDst * gpuNum + gpuRankDst;
        flow->dstId = dstId;
        // 2. 针对该gpu，随机选择一条从src到dst的可行路径，将flow分配到这条路径上。路径通常有5个节点，起点，三中间节点，终点
        std::vector<int> path;
        int srcLeafId = serverGroupNum * gpuNum + gpuRankSrc;
        int spineId =  serverGroupNum * gpuNum+ leafNum + rand() % spineNum;
        int dstLeafId = serverGroupNum * gpuNum + gpuRankDst;
        
        path = {srcId, srcLeafId, spineId, dstLeafId, dstId};

        flow->setPath(path);
    }
}


/*
目标：
    实现waterFilling函数，求出每个flow的rate
思路：
    1. 遍历两个流管理器gpuFlowManager和bgFlowManager里的所有flow，如果flow里的dataSize不为0，更新topo里的流数量  
    2. 求出gpuFlowManager里每个gpu的flow rate
        排除dataSize为0的flow，遍历每个flow的path，求出每个link上的流数量flowNum和带宽linkBW，
        rate = min(rate, linkBW / flowNum)
    3. 求出bgFlowManager里每个干扰流的flow rate
        排除dataSize为0的flow，遍历每个flow的path，求出每个link上的流数量flowNum和带宽linkBW，
        rate = min(rate, linkBW / flowNum)
    4. 将topo里的流数量清零，以便下一个时间片的更新
*/

void Network::waterFilling() {
    // 1. 遍历两个流管理器gpuFlowManager和bgFlowManager里的所有flow，如果flow里的dataSize > 0，更新topo里的流数量
    for (auto& flow : gpuFlowManager) {
        if (flow->dataSize > 0) {
            for (int i = 0; i < flow->path.size() - 1; i++) {
                topo[flow->path[i]][flow->path[i + 1]].first++;
            }
        }
    }
    for (auto& flow : bgFlowManager) {
        if (flow->dataSize > 0) {
            for (int i = 0; i < flow->path.size() - 1; i++) {
                topo[flow->path[i]][flow->path[i + 1]].first++;
            }
        }
    }

    // 2. 求出gpuFlowManager里每个gpu的flow rate，rate根据flow的path和topo里的topoBW/流数量来计算
    // 之前已经排除了空流的情况，所以这里不需要再判断flow的dataSize是否为0
    for (auto& flow : gpuFlowManager) {
        float rate = FLOAT_MAX;
        std::vector<int> path = flow->path;
        for (int i = 0; i < path.size() - 1; i++) {
            int flowNum = topo[path[i]][path[i + 1]].first;
            float linkBW = topo[path[i]][path[i + 1]].second;
            rate = std::min(rate, linkBW / flowNum);
        }
        flow->setRate(rate);
    }

    // 3. 求出bgFlowManager里每个干扰流的flow rate，rate根据flow的path和topo里的topoBW/流数量来计算
    for (auto& flow : bgFlowManager) {
        float rate = FLOAT_MAX;
        for (int i = 0; i < flow->path.size() - 1; i++) {
            int flowNum = topo[flow->path[i]][flow->path[i + 1]].first;
            float linkBW = topo[flow->path[i]][flow->path[i + 1]].second;
            rate = std::min(rate, flowNum == 0 ? linkBW : linkBW / flowNum);
        }
        flow->setRate(rate);
    }

    // 4. 将topo里的流数量清零，以便下一个时间片的更新
    for (int i = 0; i < topo.size(); i++) {
        for (int j = 0; j < topo[i].size(); j++) {
            topo[i][j].first = 0;
        }
    }
}

// setp函数的实现，执行每个server里的step函数
void Network::step(float unitTime) {
    for (auto& server : serverGroup) {
        server.step(unitTime);
    }
}

// control函数的实现，执行每个server里的control函数
void Network::control() {
    for (auto& server : serverGroup) {
        server.control();
    }
    waterFilling();
}