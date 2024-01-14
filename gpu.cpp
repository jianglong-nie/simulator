#include "gpu.h"

// GPU构造函数的实现
GPU::GPU(int rank, int dataSize, bool isFinished) {
    this->rank = rank;
    this->dataSize = dataSize;
    this->isFinished = isFinished;
}

// GPU的初始化函数，给GPU的每个成员变量都赋值
void GPU::init(int rank, int dataSize, bool isFinished) {
    this->rank = rank;
    this->dataSize = dataSize;
    this->isFinished = isFinished;
}

// 
void GPU::computing(int unitTime) {
    
}

/* 
遍历flows，对每个flow根据flow的rate，减少flow的dataSize，
dataSize = dataSize - rate * unitTime
直至flow的dataSize都为0，将不再参与计算
*/

void GPU::communication(int unitTime) {
    for (auto& flow : flows) {
        if (flow.dataSize <= 0) {
            flow.dataSize = 0;
            continue;
        }
        flow.dataSize -= flow.rate * unitTime;
    }
}

void GPU::checkFlows() {
    for (const auto& flow : flows) {
        if (flow.dataSize != 0) {
            return;
        }
    }
    isFinished = true;
}

// 
void GPU::step(int unitTime) {
    computing(unitTime);
    communication(unitTime);
}

// 
void GPU::control() {
    checkFlows();
}