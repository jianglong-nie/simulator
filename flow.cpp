#include "flow.h"

Flow::Flow(int src, int dst, int dataSize, std::string protocol) {
    this->src = src;
    this->dst = dst;
    this->dataSize = dataSize;
    this->protocol = protocol;
}

void Flow::init(int src, int dst, int dataSize, std::string protocol) {
    this->src = src;
    this->dst = dst;
    this->dataSize = dataSize;
    this->protocol = protocol;
}

void Flow::setRate(int rate) {
    this->rate = rate;
}

void Flow::setPath(std::vector<int> path) {
    this->path = path;
}