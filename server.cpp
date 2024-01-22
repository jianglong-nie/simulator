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

// step函数的实现
void Server::step(float unitTime) {
    for (auto& gpu : gpus) {
        gpu.step(unitTime);
    }
}

// control函数的实现
void Server::control(float unitTime) {
    for (auto& gpu : gpus) {
        gpu.control(unitTime);
    }
}