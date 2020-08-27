#ifndef PYSFMLNET_H
#define PYSFMLNET_H

#include <SFML/Network.hpp>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

extern sf::Packet packet;

template <typename T>
void write_data(T t) {
	packet << t;
}

template <typename T>
T read_data() {
	T t;
	packet >> t;
	return t;
}

bool init(int tcp_port, int udp_port);
void clear();
uint16_t receive_udp();
uint16_t checkUdpPacket();
uint16_t receive_tcp();
uint8_t get_tcp_code();
void sendTCP();
void sendUDP();
std::vector<uint8_t> read_state();
void send_state(uint8_t, std::vector<uint8_t>);


PYBIND11_MODULE(MODULE_NAME, m) {
  m.doc() = "python wrapper for SFML network";

	m.def("init", &init);
	m.def("checkUdpPacket", &checkUdpPacket);
	m.def("receive_tcp", &receive_tcp);
	m.def("get_tcp_code", &get_tcp_code);
	m.def("receive_udp", &receive_udp);
	m.def("sendTCP", &sendTCP);
	m.def("sendUDP", &sendUDP);
	m.def("read_state", &read_state);
	m.def("send_state", &send_state);

	m.def("write_uint8", &write_data<uint8_t>);
	m.def("write_uint16", &write_data<uint16_t>);
	m.def("write_uint32", &write_data<uint32_t>);
	m.def("write_uint64", &write_data<unsigned long long>);
	m.def("write_int8", &write_data<int8_t>);
	m.def("write_int16", &write_data<int16_t>);
	m.def("write_int32", &write_data<int32_t>);
	m.def("write_int64", &write_data<long long>);
	m.def("write_float", &write_data<float>);
	m.def("write_double", &write_data<double>);
	m.def("write_string", &write_data<std::string>);

	m.def("read_uint8", &read_data<uint8_t>);
	m.def("read_uint16", &read_data<uint16_t>);
	m.def("read_uint32", &read_data<uint32_t>);
	m.def("read_uint64", &read_data<unsigned long long>);
	m.def("read_int8", &read_data<int8_t>);
	m.def("read_int16", &read_data<int16_t>);
	m.def("read_int32", &read_data<int32_t>);
	m.def("read_int64", &read_data<long long>);
	m.def("read_float", &read_data<float>);
	m.def("read_double", &read_data<double>);
	m.def("read_string", &read_data<std::string>);
}

#endif
