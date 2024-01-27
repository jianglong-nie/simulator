#include "network.h"

// Network构造函数的实现
Network::Network() {}

// init函数的实现
void Network::init(int serverGroupNum, int gpuNum, float gpuDataSize, std::vector<std::vector<float>> NVLink, float topoBW) {
    this->serverGroupNum = serverGroupNum;
    this->gpuNum = gpuNum;
    this->gpuDataSize = gpuDataSize;
    this->NVLink = NVLink;
    this->topoBW = topoBW;
    this->nodeNum = (serverGroupNum * gpuNum) + leafNum + spineNum;

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
    实现Routing函数，根据gpuFlowManager将gpu里的flow分配到一个路径上
    针对8server，每个server里8GPU，8Leaf，8Spine的网络拓扑
    要求在分配路径时，每个链路上flow的数量不能超过2，<= 2
思路：
    1. 遍历gpuFlowManager所有的flow，确定flow所属的gpu，针对该flow的src，dst换算出在topo上的节点编号
    2. 针对该gpu，选择一条从src到dst的可行路径，将flow分配到这条路径上。路径通常有5个节点，起点，三中间节点，终点
    3. 要求在分配路径时，每个链路上flow的数量不能超过2，<= 2
*/
void Network::Routing(int gpuFlowRoutingNum) {
    // server * gpu = 8 * 8 = 64，leaf = 8，spine = 8
    std::vector<int> spineIdList = {72, 73, 74, 75, 76, 77, 78, 79};
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
        // 2. 针对该gpu，选择一条从src到dst的可行路径，将flow分配到这条路径上。路径通常有5个节点，起点，三中间节点，终点
        std::vector<int> path;
        int srcLeafId = serverGroupNum * gpuNum + gpuRankSrc;
        int dstLeafId = serverGroupNum * gpuNum + gpuRankDst;

        // 3. 要求在分配路径时，每个链路上flow的数量不能超过2，<= 1
        int spineId;
        int count = 0;
        do {
            if(count > 100) {
                break;
            }
            count++;
            spineId = spineIdList[rand() % spineNum];
        } while (topo[srcLeafId][spineId].first >= gpuFlowRoutingNum || topo[spineId][dstLeafId].first >= gpuFlowRoutingNum);
        
        
        path = {srcId, srcLeafId, spineId, dstLeafId, dstId};

        flow->setPath(path);

        // 更新topo里的流数量
        for (int i = 0; i < path.size() - 1; i++) {
            topo[path[i]][path[i + 1]].first++;
        }
    }

}

/*
目标：
    实现waterFilling函数，求出每个flow的rate
思路：
    1. 先将上一步里topo里的流数量清零，以便当前时间片的更新 
    2. 遍历两个流管理器gpuFlowManager和bgFlowManager里的所有flow，如果flow里的dataSize不为0，更新topo里的流数量  
    3. 求出gpuFlowManager里每个gpu的flow rate
        排除dataSize为0的flow，遍历每个flow的path，求出每个link上的流数量flowNum和带宽linkBW，
        rate = min(rate, linkBW / flowNum)
    4. 求出bgFlowManager里每个干扰流的flow rate
        排除dataSize为0的flow，遍历每个flow的path，求出每个link上的流数量flowNum和带宽linkBW，
        rate = min(rate, linkBW / flowNum) 
*/

void Network::waterFilling() {
    
    // 先将上一步里topo里的流数量清零，以便当前时间片的更新
    for (int i = 0; i < topo.size(); i++) {
        for (int j = 0; j < topo[i].size(); j++) {
            topo[i][j].first = 0;
        }
    }

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
    // 注意排除了空流的情况即dataSize = 0
    for (auto& flow : gpuFlowManager) {
        if (flow->dataSize > 0) {
            float rate = FLOAT_MAX;
            std::vector<int> path = flow->path;
            for (int i = 0; i < path.size() - 1; i++) {
                int flowNum = topo[path[i]][path[i + 1]].first;
                float linkBW = topo[path[i]][path[i + 1]].second;
                rate = std::min(rate, linkBW / flowNum);
            }
            flow->setRate(rate);
        }
    }

    // 3. 求出bgFlowManager里每个干扰流的flow rate，rate根据flow的path和topo里的topoBW/流数量来计算
    for (auto& flow : bgFlowManager) {
        if (flow->dataSize > 0) {
            float rate = FLOAT_MAX;
            std::vector<int> path = flow->path;
            for (int i = 0; i < path.size() - 1; i++) {
                int flowNum = topo[path[i]][path[i + 1]].first;
                float linkBW = topo[path[i]][path[i + 1]].second;
                rate = std::min(rate, linkBW / flowNum);
            }
            flow->setRate(rate);
        }
    }
}

/*
目标：
    实现dijkstra算法，该算法专门为网络中的周期性背景流量寻找一个路由路径
    我不懂你的dijkstra的算法， 请明确一下，
    你不需要用到topo[u][v].first，因为那是代表两节点之间链路上flow 的数量，
    你只需要找到topo中srcId与dstId的最短路径，topo中任意两点的路径长度为topo[u][v].second，且为float。 
    请重新修改你的dijkstra代码
*/
std::vector<int> Network::dijkstra(int srcId, int dstId) {
    int nodeNum = this->nodeNum;
    std::vector<float> dist(nodeNum, std::numeric_limits<float>::max());
    std::vector<int> prev(nodeNum, -1);
    std::vector<bool> visited(nodeNum, false);
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> pq;

    dist[srcId] = 0;
    pq.push({0, srcId});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        visited[u] = true;

        for (int v = 0; v < nodeNum; v++) {
            float weight = this->topo[u][v].second;
            if (!visited[v] && dist[u] + weight < dist[v] && weight != 0) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
                prev[v] = u;
            }
        }
    }

    // Build the shortest path from src to dst
    std::vector<int> path;
    for (int at = dstId; at != -1; at = prev[at]) {
        path.push_back(at);
    }
    std::reverse(path.begin(), path.end());

    return path;
}

// setp函数的实现，执行每个server里的step函数
void Network::step(float unitTime) {
    for (auto& server : serverGroup) {
        server.step(unitTime);
    }
    // 背景流量的step
    for (auto& flow : bgFlowManager) {
        if (flow->dataSize > 0) {
            flow->dataSize -= flow->rate * unitTime;
        }
        else {
            flow->dataSize = 0;
        }
    }
}

// control函数的实现，执行每个server里的control函数
void Network::control(float unitTime) {
    for (auto& server : serverGroup) {
        server.control(unitTime);
    }
    waterFilling();
}