#include "pySFMLnet.h"
#include <list>

sf::Packet packet;
sf::SocketSelector selector;
sf::UdpSocket udpSock;
sf::TcpSocket tcpSock;
sf::IpAddress ip;
int udp_port;


bool init(std::string ip_str, int tcp_port, int udp_port_) {
  udp_port = udp_port_;
  ip = sf::IpAddress(ip_str);
  tcpSock.connect( ip, tcp_port );
  udpSock.bind( udp_port, ip );
  selector.add(udpSock);
  return true;
}

void clear(){
  packet.clear();
}


uint16_t receive_udp() {
  if (selector.isReady(udpSock)) {
    sf::IpAddress ip;
    uint16_t udp_port;
    auto status = udpSock.receive(packet, ip, udp_port);
		if (status == sf::Socket::Done)
			return checkUdpPacket();
	}
  return 0;
}
uint16_t checkUdpPacket() {
  uint8_t packetid;
  packet >> packetid;

  if (packetid == 102) { // if Ping:
    sendUDP();           // Pong!
    return 0;
  }
  else {
    // uint16_t clientid;
		//packet >> clientid;
		return 1;
  }
  return 0;
}


uint16_t receive_tcp() {
		if (selector.isReady(tcpSock)) {
			sf::Socket::Status status = tcpSock.receive(packet);
			if (status == sf::Socket::Done){
        return 1;
      }
      // else {
      //   return 0;
      // }
	  }
  return 0;
}

uint8_t get_tcp_code(){
  uint8_t packetid;
  packet >> packetid;
  return packetid;
}

void sendTCP() {
    if (tcpSock.send(packet) != sf::Socket::Done)
      std::cout << "Error sending TCP packet to client " << ip << std::endl;
    return;
}
void sendUDP() {
      if (udpSock.send(packet, ip, udp_port) != sf::Socket::Done)
        std::cout << "Error sending UDP packet to server " << ip << std::endl;
      return;
}


///////////////////////////////////////////////////
//                Pre-made packets!
///////////////////////////////////////////////////

//read a state that was sent to us
std::vector<uint8_t> read_state(){
  // uint8_t signal;
  // std::size_t count;
  uint8_t signal;
  uint16_t count;
  packet >> signal;
  packet >> count;
  std::vector<uint8_t> state(count);
  for (auto &x : state){
    packet >> x;
  }
  return state;
}
//send our state to server
void send_state(uint8_t id, std::vector<uint8_t> state){
  packet.clear();
  std::size_t n = state.size();
  packet << (uint8_t)100 << id << (uint16_t)n;
  for (auto x : state){
    packet << x;
  }
  sendUDP();
}
