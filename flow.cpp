#include "flow.h"

Flow::Flow(std::pair<int, int> src, std::pair<int, int> dst, float dataSize, std::string protocol) {
    this->src = src;
    this->dst = dst;
    this->dataSize = dataSize;
    this->protocol = protocol;
}

void Flow::init(std::pair<int, int> src, std::pair<int, int> dst, float dataSize, std::string protocol) {
    this->src = src;
    this->dst = dst;
    this->dataSize = dataSize;
    this->protocol = protocol;
}

void Flow::setRate(float rate) {
    this->rate = rate;
}

void Flow::setPath(std::vector<int> path) {
    this->path = path;
}