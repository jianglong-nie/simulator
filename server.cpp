#include "server.h"

// Server构造函数的实现
Server::Server(int id, int gpuNum, float gpuDataSize, std::vector<std::vector<float>> NVLink) {
    this->id = id;
    this->gpuNum = gpuNum;
    this->NVLink = NVLink;
    for (int i = 0; i < gpuNum; i++) {
        gpus.push_back(GPU(i, gpuDataSize, false));
    }
}

// Server的初始化函数，给Server的每个成员变量都赋值
void Server::init(int id, int gpuNum, float gpuDataSize, std::vector<std::vector<float>> NVLink) {
    this->id = id;
    this->gpuNum = gpuNum;
    this->NVLink = NVLink;
    for (int i = 0; i < gpuNum; i++) {
        gpus.push_back(GPU(i, gpuDataSize, false));
    }
}

// 对滑动窗口的初始化
// 考虑对该server下的所有GPU都进行一样的操作
void Server::initWindow(float len, float Cnvl, float Cnet, float Wnvl, float Wnet, float alpha, float delta) {
    this->len = len;
    this->Cnvl = Cnvl;
    this->Cnet = Cnet;
    this->alpha = alpha;
    this->delta = delta;
    // 开始判断
    
    // 在这两种情况下，分配给NVLink和Net的数据都固定好了，窗口不需要滑动了
    if (this->len <= std::min(Wnvl, Wnet)) { // len < Wnvl && len < Wnet
        this->Wnvl = this->len;
        this->Wnet = 0;
        for (auto& gpu: gpus) {
            gpu.flows[0].dataSize = this->Wnvl;
            gpu.dataSize = 0;
        }
        return;
    }
    else if (len <= Wnvl + Wnet) { // len < Wnet < Wnvl
        this->Wnvl = Wnvl;
        this->Wnet = len - Wnvl;
        for (auto& gpu: gpus) {
            gpu.flows[0].dataSize = this->Wnvl;
            gpu.flows[1].dataSize = this->Wnet;
            gpu.dataSize = 0;
        }
        return;
    }
    this->Wnet = Wnet;
    this->Wnvl = Wnvl;
    left = this->Wnvl;
    right = len - this->Wnet;
    // 分配一个Cnvl和Cnet给flows[0]和flows[1]
    for (auto& gpu: gpus) {
        gpu.sendChunk("NVLink");
        gpu.sendChunk("Net");
    }
    return;
}

// 判断该server下的所有GPU是否都完成了
bool Server::allRanksFinish(std::string protocol) {
    for (auto& gpu : gpus) {
        if (!gpu.rankFinish(protocol)) {
            return false;
        }
    }
    return true;
}

void Server::computing(float unitTime) {
    if (left == -1 && right == -1) {
        return;
    }
    else if (0 < left && left < right) {
        if (allRanksFinish("NVLink")) { //
            left += Cnvl; // 考虑left与right临近时，可能有一定重叠，但是对仿真结果影响不大
            for (auto& gpu : gpus) {
                gpu.sendChunk("NVLink");
            }
        }
        if (left < right && allRanksFinish("Net")) {
            this->timeChunkNow = FLOAT_MAX;
            for (auto& gpu : gpus) {
                if (gpu.timeChunkNow < this->timeChunkNow) {
                    this->timeChunkNow = gpu.timeChunkNow;
                }
            }
            if (timeChunkNow - timeChunkLast < delta) { // 假设delta是你已经定义的变量
                Wnet += alpha * Cnet; // 假设alpha是你已经定义的变量
                right -= (alpha + 1) * Cnet;
                for (auto& gpu : gpus) {
                    gpu.sendChunk("Net");
                }
            } else if (Wnet > Cnet) {
                Wnet -= Cnet;
            }
            else {
                right -= Cnet;
                for (auto& gpu : gpus) {
                    gpu.sendChunk("Net");
                }
            }
            this->timeChunkLast = this->timeChunkNow;
            this->timeChunkNow = 0;
            for (auto& gpu : gpus) {
                gpu.timeChunkNow = 0;
            }
        }
    }
    else if(left >= right && right > 0){
        for (auto&gpu : gpus) {
            gpu.flows[0].dataSize -= Cnvl; // 去掉left++时的sendChunk, left后退一格chunk
            gpu.flows[0].chunkDataSize -= Cnvl;

            float nvlDataSize = right - gpu.flows[0].chunkDataSize;
            float netDataSize = (len - right) - gpu.flows[1].chunkDataSize;
            // 考虑如果left和right如果重叠，那么right保持不变，left后退一点到right的值
        
            gpu.dataSize -= nvlDataSize;
            gpu.dataSize -= netDataSize;

            gpu.flows[0].dataSize += nvlDataSize;
            gpu.flows[1].dataSize += netDataSize;
        }
        left = -1;
        right = -1;
    }
}

void Server::computingNonCC(float unitTime) {
    if (left == -1 && right == -1) {
        return;
    }
    else if (0 < left && left < right) {
        if (allRanksFinish("NVLink")) { //
            left += Cnvl; // 考虑left与right临近时，可能有一定重叠，但是对仿真结果影响不大
            for (auto& gpu : gpus) {
                gpu.sendChunk("NVLink");
            }
        }
        if (left < right && allRanksFinish("Net")) {
            Wnet += Cnet;
            right -= (1) * Cnet;
            for (auto& gpu : gpus) {
                gpu.sendChunk("Net");
            }
        }
    }
    else if(left >= right && right > 0){
        for (auto&gpu : gpus) {
            gpu.flows[0].dataSize -= Cnvl; // 去掉left++时的sendChunk, left后退一格chunk
            gpu.flows[0].chunkDataSize -= Cnvl;

            float nvlDataSize = right - gpu.flows[0].chunkDataSize;
            float netDataSize = (len - right) - gpu.flows[1].chunkDataSize;
            // 考虑如果left和right如果重叠，那么right保持不变，left后退一点到right的值
        
            gpu.dataSize -= nvlDataSize;
            gpu.dataSize -= netDataSize;

            gpu.flows[0].dataSize += nvlDataSize;
            gpu.flows[1].dataSize += netDataSize;
        }
        left = -1;
        right = -1;
    }
}


// step函数的实现
void Server::step(float unitTime) {
    for (auto& gpu : gpus) {
        gpu.step(unitTime);
    }
}

// control函数的实现
void Server::control(float unitTime) {
    //computing(unitTime);
    computingNonCC(unitTime);
    for (auto& gpu : gpus) {
        gpu.control(unitTime);
    }
}