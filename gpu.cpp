#include "gpu.h"

// GPU构造函数的实现
GPU::GPU(int rank, float dataSize, bool isFinished) {
    this->rank = rank;
    this->dataSize = dataSize;
    this->isFinished = isFinished;
}

// GPU的初始化函数，给GPU的每个成员变量都赋值
void GPU::init(int rank, float dataSize, bool isFinished) {
    this->rank = rank;
    this->dataSize = dataSize;
    this->isFinished = isFinished;
    this->len = dataSize;
}

// 对滑动窗口的初始化
void GPU::initWindow(float len, float Cnvl, float Cnet, float Wnvl, float Wnet, float alpha, float delta) {
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
        flows[0].dataSize = this->Wnvl;
        this->dataSize = 0;
        return;
    }
    else if (len <= Wnvl + Wnet) { // len < Wnet < Wnvl
        this->Wnvl = Wnvl;
        this->Wnet = len - Wnvl;
        flows[0].dataSize = this->Wnvl;
        flows[1].dataSize = this->Wnet;
        this->dataSize = 0;
        return;
    }
    this->Wnet = Wnet;
    this->Wnvl = Wnvl;
    left = this->Wnvl;
    right = len - this->Wnet;
    // 分配一个Cnvl和Cnet给flows[0]和flows[1]
    sendChunk("NVLink");
    sendChunk("Net");
    return;
}

bool GPU::rankFinish(std::string protocol) {
    if (protocol == "NVLink" && flows[0].dataSize <= 0) {
        return true;
    }
    else if (protocol == "Net" && flows[1].dataSize <= 0) {
        return true;
    }
    else {
        return false;
    }
}

void GPU::sendChunk(std::string protocol) {
    if (protocol == "NVLink") {
        if (dataSize > Cnvl) {
            flows[0].dataSize += Cnvl;
            flows[0].chunkDataSize += Cnvl;
            dataSize -= Cnvl;
        }
        else {
            flows[0].dataSize += dataSize;
            flows[0].chunkDataSize += dataSize;
            dataSize = 0;
        }
    }
    else if (protocol == "Net") {
        if (dataSize > Cnet) {
            flows[1].dataSize += Cnet;
            flows[1].chunkDataSize += Cnet;
            dataSize -= Cnet;
        }
        else {
            flows[1].dataSize += dataSize;
            flows[1].chunkDataSize += dataSize;
            dataSize = 0;
        }
    }
}

// 计算函数，根据当前的left和right，判断是否需要发送chunk给NVLink或者Network
void GPU::computing(float unitTime) {
    if (left == -1 && right == -1) {
        return;
    }
    else if (0 < left && left < right) {
        if (rankFinish("NVLink")) { //
            left += Cnvl; // 考虑left与right临近时，可能有一定重叠，但是对仿真结果影响不大
            sendChunk("NVLink");
        }
        if (left < right && rankFinish("Net")) {
            if (timeChunkNow - timeChunkLast < delta) { // 假设delta是你已经定义的变量
                Wnet += alpha * Cnet; // 假设alpha是你已经定义的变量
                right -= (alpha + 1) * Cnet;
                sendChunk("Net");
            } else if (Wnet > Cnet) {
                Wnet -= Cnet;
                //return;
            }
            else {
                right -= Cnet;
                sendChunk("Net");
            }
            timeChunkLast = timeChunkNow;
            timeChunkNow = 0;
        }
    }
    else if(left >= right && right > 0){
        flows[0].dataSize -= Cnvl; // 去掉left++时的sendChunk, left后退一格chunk
        flows[0].chunkDataSize -= Cnvl;
        // 考虑如果left和right如果重叠，那么right保持不变，left后退一点到right的值
        float nvlDataSize = right - flows[0].chunkDataSize;
        
        float netDataSize = (len - right) - flows[1].chunkDataSize;

        dataSize -= nvlDataSize;
        dataSize -= netDataSize;

        flows[0].dataSize += nvlDataSize;
        flows[1].dataSize += netDataSize;

        left = -1;
        right = -1;
    }
}

/* 
遍历flows，对每个flow根据flow的rate，减少flow的dataSize，
dataSize = dataSize - rate * unitTime
直至flow的dataSize都为0，将不再参与计算
*/

// 通信函数，根据flows的rate，减少flows的dataSize, 但是dataSize减少为0，不会为负数
void GPU::communication(float unitTime) {
    float epsilon = 1e-5;
    for (auto& flow : flows) {
        if (flow.dataSize > 0) {
            if (flow.dataSize > flow.rate * unitTime) {
                flow.dataSize -= flow.rate * unitTime;
                flow.sentDataSize += flow.rate * unitTime;
            }
            else {
                flow.sentDataSize += flow.dataSize;
                flow.dataSize = 0;
            }
            flow.completionTime += unitTime;
        }
    }
    timeChunkNow += unitTime;
}

// 如果dataSize为0，且每个flow的dataSize都为0，说明GPU已经结束了工作
void GPU::isWorkFinished() {
    float epsilon = 1e-5;
    if (dataSize > epsilon) {
        return;
    }
    for (const auto& flow : flows) {
        if (flow.dataSize > epsilon) {
            return;
        }
    }
    isFinished = true;
}

// 
void GPU::step(float unitTime) {
    communication(unitTime);
}

// 
void GPU::control(float unitTime) {
    computing(unitTime);
    isWorkFinished();
}